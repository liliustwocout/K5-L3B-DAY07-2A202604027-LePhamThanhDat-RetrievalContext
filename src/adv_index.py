"""Full 150-page multimodal PDF indexer for Day 7 Advanced Challenge (NOAA Arctic Report Card 2025).

Extracts:
1. Text chunks (section-aware paragraph chunking up to 1200 chars) across all 150 pages.
2. Structured Markdown tables + cropped PNGs via `pymupdf.Page.find_tables()`.
3. Visual regions (embedded raster charts/photos/maps + non-table vector diagrams) + cropped PNGs
   + Gemini 3.5 Flash Lite structured vision transcription.
4. Embeddings (`gemini-embedding-001`, 768 dims) stored idempotently in PostgreSQL `adv_chunks` (`pgvector`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import psycopg
import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from retrievers import _extract_retry_seconds, _is_retryable_api_error, _l2_normalize, _to_pgvector, normalize_text

ROOT = Path(__file__).resolve().parents[1]
PDF_URL = "https://arctic.noaa.gov/wp-content/uploads/2025/12/ArcticReportCard_full_report2025.pdf"
DEFAULT_PDF_PATH = ROOT / "data" / "adv_source" / "ArcticReportCard_full_report2025.pdf"

_RUNNING_HEADER_RE = re.compile(
    r"NOAA Technical Report OAR ARC\s*(?:;\s*[\d-]+)?\s*Arctic Report Card 2025\s*\d*",
    flags=re.IGNORECASE,
)
_CAPTION_START_RE = re.compile(r"^(?:Fig\.|Figure|Table)\s*\d+", flags=re.IGNORECASE)


def ensure_pdf(pdf_path: Path, url: str = PDF_URL) -> dict[str, Any]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            pdf_path.write_bytes(resp.read())
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return {
        "title": "Arctic Report Card 2025: Twenty years of tracking rapid Arctic warming and change",
        "editors": "M. L. Druckenmiller, R. L. Thoman, and T. A. Moon",
        "publisher": "NOAA Technical Report OAR ARC (December 2025)",
        "doi": "10.25923/nrzf-j897",
        "url": url,
        "license": "Public Domain (U.S. Government Work, 17 U.S.C. § 105)",
        "local_path": str(pdf_path.relative_to(ROOT)),
        "size_bytes": pdf_path.stat().st_size,
        "sha256": sha256,
    }


def _chapter_map(doc: pymupdf.Document) -> dict[int, str]:
    """Map 1-based page numbers to level-1 chapter titles from PDF TOC."""
    toc = [(lvl, title, p) for lvl, title, p in doc.get_toc() if lvl == 1]
    out: dict[int, str] = {}
    cur = "Frontmatter"
    toc_idx = 0
    for pno in range(1, len(doc) + 1):
        while toc_idx < len(toc) and toc[toc_idx][2] <= pno:
            cur = toc[toc_idx][1]
            toc_idx += 1
        out[pno] = cur if pno <= 144 else "Authors and Affiliations"
    return out


def _rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def _table_to_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    cleaned_rows: list[list[str]] = []
    max_cols = max(len(r) for r in rows)
    for r in rows:
        cells = [normalize_text(str(c or "")).replace("|", "/") for c in r]
        while len(cells) < max_cols:
            cells.append("")
        cleaned_rows.append(cells)
    header = cleaned_rows[0]
    sep = ["---"] * max_cols
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in cleaned_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _nearest_caption(
    blocks: list[tuple[float, float, float, float, str]],
    bbox: tuple[float, float, float, float],
    kind: str,
) -> str:
    """Find the closest Fig./Table caption block on the same page near bbox."""
    prefix = ("table",) if kind == "table" else ("fig.", "figure")
    best_dist = 1e9
    best_txt = ""
    for bx0, by0, bx1, by1, btxt in blocks:
        norm = normalize_text(btxt)
        low = norm.casefold()
        if not any(low.startswith(p) for p in prefix):
            continue
        # Vertical gap between caption block and figure/table bbox
        if by1 <= bbox[1]:
            dist = bbox[1] - by1
        elif by0 >= bbox[3]:
            dist = by0 - bbox[3]
        else:
            dist = 0.0
        if dist < best_dist and dist <= 220.0:
            best_dist = dist
            best_txt = norm
    return best_txt


def _crop_to_png(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    *,
    pad_x: float = 12.0,
    pad_top: float = 14.0,
    pad_bottom: float = 14.0,
    dpi: int = 160,
) -> tuple[float, float, float, float]:
    pr = page.rect
    clip = pymupdf.Rect(
        max(pr.x0, bbox[0] - pad_x),
        max(pr.y0, bbox[1] - pad_top),
        min(pr.x1, bbox[2] + pad_x),
        min(pr.y1, bbox[3] + pad_bottom),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    pix.save(str(out_path))
    return (round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1))


def _describe_visual_with_gemini(
    client: genai.Client,
    model_name: str,
    crop_path: Path,
    *,
    page: int,
    section: str,
    caption: str,
    cache: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    raw_bytes = crop_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    key = f"{model_name}:{digest}"
    if key in cache:
        return cache[key]

    prompt = (
        "You are indexing a visual figure/chart/photo/diagram from the NOAA Arctic Report Card 2025.\n"
        "Inspect the image carefully and extract ONLY directly visible facts (never guess unseen numbers).\n"
        f"Page: {page} | Section: {section}\n"
        f"Nearby page caption: {caption or '(none)'}\n\n"
        "Return a single JSON object with keys:\n"
        '- "visual_type": one of "chart", "photo", "diagram", "map", "decorative"\n'
        '- "title": short title of what the figure shows\n'
        '- "axes_units_legend": exact axis titles, units, scale ranges, and legend series (or "N/A" for photos)\n'
        '- "summary_and_values": thorough factual summary of visible labels, key numerical values, peaks/lows, '
        "trends, spatial regions, or photographic subjects/details."
    )
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=raw_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            txt = (resp.text or "").strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)
            parsed = json.loads(txt)
            cache[key] = parsed
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.35)
            return parsed
        except Exception as exc:
            if isinstance(exc, json.JSONDecodeError):
                if attempt == max_attempts:
                    parsed = {
                        "visual_type": "diagram",
                        "title": caption[:120] or f"Figure on page {page}",
                        "axes_units_legend": "N/A",
                        "summary_and_values": caption or "Visual region indexed by crop only.",
                    }
                    cache[key] = parsed
                    return parsed
                time.sleep(1.5)
                continue
            if not _is_retryable_api_error(exc) or attempt == max_attempts:
                raise
            time.sleep(_extract_retry_seconds(str(exc), attempt))
    raise RuntimeError("Failed to describe visual region")


def _split_text_blocks(
    blocks: list[tuple[float, float, float, float, str]],
    max_size: int = 1200,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Group paragraph text blocks up to max_size while preserving union bbox."""
    chunks: list[tuple[str, tuple[float, float, float, float]]] = []
    buf_texts: list[str] = []
    buf_box: list[float] | None = None

    def flush() -> None:
        nonlocal buf_texts, buf_box
        if buf_texts and buf_box is not None:
            chunks.append(("\n\n".join(buf_texts), (round(buf_box[0], 1), round(buf_box[1], 1), round(buf_box[2], 1), round(buf_box[3], 1))))
        buf_texts = []
        buf_box = None

    for bx0, by0, bx1, by1, raw in blocks:
        clean = _RUNNING_HEADER_RE.sub("", normalize_text(raw)).strip()
        if not clean:
            continue
        if len(clean) > max_size:
            flush()
            for i in range(0, len(clean), max_size):
                piece = clean[i : i + max_size].strip()
                if piece:
                    chunks.append((piece, (round(bx0, 1), round(by0, 1), round(bx1, 1), round(by1, 1))))
            continue
        candidate = ("\n\n".join(buf_texts + [clean])) if buf_texts else clean
        if len(candidate) <= max_size:
            buf_texts.append(clean)
            if buf_box is None:
                buf_box = [bx0, by0, bx1, by1]
            else:
                buf_box[0] = min(buf_box[0], bx0)
                buf_box[1] = min(buf_box[1], by0)
                buf_box[2] = max(buf_box[2], bx1)
                buf_box[3] = max(buf_box[3], by1)
        else:
            flush()
            buf_texts = [clean]
            buf_box = [bx0, by0, bx1, by1]
    flush()
    return chunks


def embed_documents_batch(
    client: genai.Client,
    texts: list[str],
    *,
    model_name: str = "gemini-embedding-001",
    dimensions: int = 768,
    cache_path: Path,
    batch_size: int = 30,
) -> list[list[float]]:
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    results: list[list[float] | None] = [None] * len(texts)
    missing_idx: list[int] = []
    for idx, txt in enumerate(texts):
        key = f"{model_name}:{dimensions}:RETRIEVAL_DOCUMENT:{hashlib.sha256(txt.encode('utf-8')).hexdigest()}"
        if key in cache:
            results[idx] = cache[key]
        else:
            missing_idx.append(idx)

    for start in range(0, len(missing_idx), batch_size):
        batch_indices = missing_idx[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_indices]
        for attempt in range(1, 6):
            try:
                resp = client.models.embed_content(
                    model=model_name,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=dimensions,
                    ),
                )
                for i, emb in zip(batch_indices, resp.embeddings):
                    vec = _l2_normalize(list(emb.values))
                    results[i] = vec
                    key = f"{model_name}:{dimensions}:RETRIEVAL_DOCUMENT:{hashlib.sha256(texts[i].encode('utf-8')).hexdigest()}"
                    cache[key] = vec
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                break
            except Exception as exc:
                if not _is_retryable_api_error(exc) or attempt == 5:
                    raise
                time.sleep(_extract_retry_seconds(str(exc), attempt))
        time.sleep(0.5)

    return [vec for vec in results if vec is not None]


def upsert_adv_chunks(database_url: str, chunks: list[dict[str, Any]], vectors: list[list[float]], dimensions: int = 768) -> None:
    with psycopg.connect(database_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS adv_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    modality TEXT NOT NULL,
                    visual_type TEXT,
                    section TEXT,
                    crop_path TEXT,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({dimensions}) NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_adv_chunks_page ON adv_chunks (page);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_adv_chunks_modality ON adv_chunks (modality);")
            keep_ids = [c["id"] for c in chunks]
            cur.execute("DELETE FROM adv_chunks WHERE NOT (chunk_id = ANY(%s));", (keep_ids,))
            rows = [
                (
                    c["id"],
                    c["source_id"],
                    c["source"],
                    int(c["page"]),
                    c["modality"],
                    c.get("visual_type"),
                    c.get("section"),
                    c.get("crop_path"),
                    c["text"],
                    json.dumps(c["metadata"], ensure_ascii=False),
                    _to_pgvector(vec),
                )
                for c, vec in zip(chunks, vectors)
            ]
            cur.executemany(
                """
                INSERT INTO adv_chunks (
                    chunk_id, source_id, source, page, modality, visual_type,
                    section, crop_path, text, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    source = EXCLUDED.source,
                    page = EXCLUDED.page,
                    modality = EXCLUDED.modality,
                    visual_type = EXCLUDED.visual_type,
                    section = EXCLUDED.section,
                    crop_path = EXCLUDED.crop_path,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                rows,
            )
        conn.commit()


def build_index(pdf_path: Path, out_dir: Path, *, vision_model: str) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not api_key or not database_url:
        raise RuntimeError("GEMINI_API_KEY and DATABASE_URL are required in .env")

    manifest = ensure_pdf(pdf_path)
    doc = pymupdf.open(pdf_path)
    chapters = _chapter_map(doc)

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=45000))
    crops_dir = out_dir / "crops"
    vision_cache_path = ROOT / "results" / "adv_index" / ".vision_cache.json"
    vision_cache: dict[str, Any] = {}
    if vision_cache_path.exists():
        vision_cache = json.loads(vision_cache_path.read_text(encoding="utf-8"))

    all_chunks: list[dict[str, Any]] = []
    skipped_items: list[dict[str, Any]] = []
    modality_counts = {"text": 0, "table": 0, "visual": 0}
    visual_type_counts: dict[str, int] = {}

    for pno in range(1, len(doc) + 1):
        page = doc[pno - 1]
        section = chapters[pno]
        raw_blocks = [
            (b[0], b[1], b[2], b[3], b[4])
            for b in page.get_text("blocks", sort=True)
            if b[6] == 0 and normalize_text(b[4])
        ]

        # Page 1 is the cover graphic; log cover metadata explicitly and index its text.
        if pno == 1:
            skipped_items.append(
                {
                    "page": 1,
                    "item": "cover_artwork",
                    "reason": "Decorative front cover illustration; text metadata indexed separately.",
                    "impact": "None on report content retrieval.",
                }
            )

        # 1. Text chunks across the entire page
        text_pieces = _split_text_blocks(raw_blocks, max_size=1200)
        for idx, (txt, bbox) in enumerate(text_pieces):
            cid = f"arc2025:p{pno:03d}:text:{idx:03d}"
            all_chunks.append(
                {
                    "id": cid,
                    "source_id": "arc-2025",
                    "source": "ArcticReportCard_full_report2025.pdf",
                    "page": pno,
                    "modality": "text",
                    "visual_type": None,
                    "section": section,
                    "crop_path": None,
                    "text": txt,
                    "metadata": {
                        "section": section,
                        "modality": "text",
                        "bbox": list(bbox),
                        "chunk_index": idx,
                    },
                }
            )
            modality_counts["text"] += 1

        if pno == 1:
            continue

        # 2. Tables via PyMuPDF find_tables()
        table_boxes: list[tuple[float, float, float, float]] = []
        found_tables = page.find_tables().tables
        for t_idx, tab in enumerate(found_tables):
            rows = tab.extract()
            md = _table_to_markdown(rows)
            if not md:
                continue
            tbbox = tuple(float(x) for x in tab.bbox)
            table_boxes.append(tbbox)
            cap = _nearest_caption(raw_blocks, tbbox, "table")
            crop_rel = f"data/adv_index/crops/p{pno:03d}_table_{t_idx:02d}.png"
            clipped = _crop_to_png(page, tbbox, ROOT / crop_rel, pad_x=10, pad_top=34, pad_bottom=12, dpi=180)
            cid = f"arc2025:p{pno:03d}:table:{t_idx:02d}"
            txt = f"[TABLE] Page {pno} ({section})\nCaption: {cap or '(no caption)'}\n{md}"
            all_chunks.append(
                {
                    "id": cid,
                    "source_id": "arc-2025",
                    "source": "ArcticReportCard_full_report2025.pdf",
                    "page": pno,
                    "modality": "table",
                    "visual_type": "table",
                    "section": section,
                    "crop_path": crop_rel,
                    "text": txt,
                    "metadata": {
                        "section": section,
                        "modality": "table",
                        "caption": cap,
                        "bbox": list(clipped),
                        "rows": len(rows),
                    },
                }
            )
            modality_counts["table"] += 1

        # 3. Visual regions: substantial raster images + non-table vector drawing clusters
        visual_boxes: list[tuple[float, float, float, float]] = []
        for info in page.get_images(full=True):
            xref, w, h = info[0], info[2], info[3]
            if w < 180 or h < 120:
                continue
            for r in page.get_image_rects(xref):
                if r.width >= 100 and r.height >= 60:
                    cand = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                    if not any(_rect_iou(cand, existing) > 0.5 for existing in visual_boxes + table_boxes):
                        visual_boxes.append(cand)

        if hasattr(page, "cluster_drawings"):
            for d in page.cluster_drawings():
                if d.width >= 120 and d.height >= 90:
                    cand = (float(d.x0), float(d.y0), float(d.x1), float(d.y1))
                    if not any(_rect_iou(cand, existing) > 0.3 for existing in visual_boxes + table_boxes):
                        visual_boxes.append(cand)

        visual_boxes.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for v_idx, vbbox in enumerate(visual_boxes):
            cap = _nearest_caption(raw_blocks, vbbox, "fig")
            crop_rel = f"data/adv_index/crops/p{pno:03d}_fig_{v_idx:02d}.png"
            clipped = _crop_to_png(page, vbbox, ROOT / crop_rel, pad_x=14, pad_top=14, pad_bottom=16, dpi=160)
            desc = _describe_visual_with_gemini(
                client,
                vision_model,
                ROOT / crop_rel,
                page=pno,
                section=section,
                caption=cap,
                cache=vision_cache,
                cache_path=vision_cache_path,
            )
            vtype = str(desc.get("visual_type") or "diagram").strip().lower()
            if vtype == "decorative" or (pno == 2 and "20th" in str(desc.get("title", "")).lower() and not cap):
                skipped_items.append(
                    {
                        "page": pno,
                        "item": crop_rel,
                        "reason": f"Decorative banner/logo ({desc.get('title')}); excluded from retrieval index.",
                        "impact": "Prevents decorative graphic pollution.",
                    }
                )
                continue
            visual_type_counts[vtype] = visual_type_counts.get(vtype, 0) + 1
            cid = f"arc2025:p{pno:03d}:fig:{v_idx:02d}"
            txt = (
                f"[VISUAL:{vtype.upper()}] Page {pno} ({section})\n"
                f"Caption: {cap or '(no caption)'}\n"
                f"Title: {desc.get('title', '')}\n"
                f"Axes/Units/Legend: {desc.get('axes_units_legend', 'N/A')}\n"
                f"Details: {desc.get('summary_and_values', '')}"
            )
            all_chunks.append(
                {
                    "id": cid,
                    "source_id": "arc-2025",
                    "source": "ArcticReportCard_full_report2025.pdf",
                    "page": pno,
                    "modality": "visual",
                    "visual_type": vtype,
                    "section": section,
                    "crop_path": crop_rel,
                    "text": txt,
                    "metadata": {
                        "section": section,
                        "modality": "visual",
                        "visual_type": vtype,
                        "caption": cap,
                        "bbox": list(clipped),
                        "vision_description": desc,
                    },
                }
            )
            modality_counts["visual"] += 1
        if pno % 25 == 0:
            print(f"Processed page {pno}/{len(doc)} | chunks={len(all_chunks)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Embed and upsert into pgvector adv_chunks
    vectors = embed_documents_batch(
        client,
        [c["text"] for c in all_chunks],
        cache_path=ROOT / "results" / "adv_index" / ".adv_doc_embed_cache.json",
    )
    upsert_adv_chunks(database_url, all_chunks, vectors, dimensions=768)

    log = {
        "document": manifest,
        "total_pages": len(doc),
        "pages_indexed": sorted({c["page"] for c in all_chunks}),
        "total_chunks": len(all_chunks),
        "modality_counts": modality_counts,
        "visual_type_counts": visual_type_counts,
        "skipped_items": skipped_items,
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
        "vision_indexer_model": vision_model,
        "vector_table": "adv_chunks",
    }
    (out_dir / "ingestion_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_pages": len(doc), "total_chunks": len(all_chunks), "modality_counts": modality_counts, "visual_type_counts": visual_type_counts}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index 150-page NOAA Arctic Report Card 2025 multimodally")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "adv_index")
    parser.add_argument("--vision-model", default="gemini-3.5-flash-lite")
    args = parser.parse_args()
    build_index(args.pdf, args.out_dir, vision_model=args.vision_model)


if __name__ == "__main__":
    main()
