"""Evaluate retrieval methods and context-packing configurations on dev or held-out test."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from retrievers import (
    BM25Retriever,
    CorpusNeighborStore,
    CrossFeatureReranker,
    GeminiQueryEmbedder,
    GeminiReranker,
    HybridRRFRetriever,
    PgVectorDenseRetriever,
    RerankedRetriever,
    TfidfRetriever,
    normalize_text,
    pack_context_budget,
)

ROOT = Path(__file__).resolve().parents[1]


def chunk_hits_gold(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Return True if normalized gold evidence appears inside chunk on matching source_id and page."""
    c_norm = normalize_text(chunk["text"])
    return any(
        chunk["source_id"] == g["source_id"]
        and int(chunk["page"]) == int(g["page"])
        and normalize_text(g["evidence"]) in c_norm
        for g in gold
    )


def context_hits_gold(
    question: dict[str, Any],
    context: str,
    packed_chunks: list[dict[str, Any]],
) -> bool:
    """Verify whether the final packed context contains all required evidence and matching source/page."""
    if not context.strip() or not packed_chunks:
        return False
    gold = question["gold"]
    has_gold_location = any(
        c["source_id"] == g["source_id"] and int(c["page"]) == int(g["page"])
        for c in packed_chunks
        for g in gold
    )
    if not has_gold_location:
        return False

    ctx_norm = normalize_text(context)
    required_spans = question.get("required_spans")
    if required_spans:
        return all(normalize_text(span) in ctx_norm for span in required_spans)
    return any(chunk_hits_gold(c, gold) for c in packed_chunks)


CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "tfidf_b800": {
        "method": "tfidf",
        "top_k": 3,
        "budget": 800,
        "neighbor_window": 0,
        "description": "Lexical TF-IDF cosine with metadata filter, top_k=3, budget=800 chars",
    },
    "dense_b800": {
        "method": "dense",
        "top_k": 3,
        "budget": 800,
        "neighbor_window": 0,
        "description": "Baseline: Day 6 pgvector Dense (gemini-embedding-001, 768d) + SQL filter, top_k=3, budget=800 chars",
    },
    "bm25_b800": {
        "method": "bm25",
        "top_k": 3,
        "budget": 800,
        "neighbor_window": 0,
        "description": "Method 2: Okapi BM25 lexical + metadata filter, top_k=3, budget=800 chars",
    },
    "hybrid_rrf_b800": {
        "method": "hybrid_rrf",
        "candidate_k": 12,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 800,
        "neighbor_window": 0,
        "description": "Method 3: Hybrid RRF (Dense + BM25, fetch_k=12, k=60), top_k=3, budget=800 chars",
    },
    "hybrid_rrf_b2400": {
        "method": "hybrid_rrf",
        "candidate_k": 12,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 2400,
        "neighbor_window": 0,
        "description": "Context budget tuning: Hybrid RRF top_k=3 with budget=2400 chars (prevents dropping >740-char Strategy C chunks)",
    },
    "hybrid_rrf_b3600": {
        "method": "hybrid_rrf",
        "candidate_k": 12,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 3600,
        "neighbor_window": 0,
        "description": "Improvement 1 (Context Budget): Hybrid RRF top_k=3 with budget=3600 chars (fits all top-3 Strategy C chunks)",
    },
    "hybrid_rrf_neighbor_b4800": {
        "method": "hybrid_rrf",
        "candidate_k": 12,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 4800,
        "neighbor_window": 1,
        "neighbor_anchors": 1,
        "description": "Improvement 2 (Selected): Hybrid RRF (top_k=3) + Rank-1 Adjacent Chunk Expansion (+1/-1) + budget=4800 chars",
    },
    "hybrid_cross_rerank_b3600": {
        "method": "hybrid_cross_rerank",
        "candidate_k": 12,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 3600,
        "neighbor_window": 0,
        "description": "Method 4 / Improvement 1b: Hybrid RRF (fetch_k=12) + Cross-Feature Reranker (dense+BM25+bigram+IDs) -> top_k=3, budget=3600 chars",
    },
    "hybrid_cross_rerank_neighbor_b4200": {
        "method": "hybrid_cross_rerank",
        "candidate_k": 12,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 4200,
        "neighbor_window": 1,
        "neighbor_anchors": 2,
        "description": "Improvement 2 (Context Engineering): Hybrid + Cross-Feature Rerank + Adjacent Chunk Expansion (window=1, anchors=2), budget=4200 chars",
    },
    "hybrid_rerank_b2400": {
        "method": "hybrid_rerank",
        "candidate_k": 10,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 2400,
        "neighbor_window": 0,
        "description": "Improvement 1: Hybrid RRF (fetch_k=10) + Gemini 3.6 Flash listwise reranker -> top_k=3, budget=2400 chars",
    },
    "hybrid_rerank_neighbor_b3200": {
        "method": "hybrid_rerank",
        "candidate_k": 10,
        "rrf_k": 60,
        "top_k": 3,
        "budget": 3200,
        "neighbor_window": 1,
        "neighbor_anchors": 2,
        "description": "Improvement 2: Hybrid RRF + Gemini 3.6 Flash Rerank + Adjacent Chunk Expansion (window=1, anchors=2), budget=3200 chars",
    },
}


def build_retriever(
    cfg: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    database_url: str,
    cache_path: Path,
):
    method = cfg["method"]
    embedder = None
    reranker = None

    if method == "tfidf":
        retriever = TfidfRetriever(chunks)
    elif method == "bm25":
        retriever = BM25Retriever(chunks)
    elif method == "dense":
        embedder = GeminiQueryEmbedder(cache_path=cache_path)
        retriever = PgVectorDenseRetriever(database_url, embedder, db_strategy="C", dimensions=768)
    elif method == "hybrid_rrf":
        embedder = GeminiQueryEmbedder(cache_path=cache_path)
        dense = PgVectorDenseRetriever(database_url, embedder, db_strategy="C", dimensions=768)
        bm25 = BM25Retriever(chunks)
        retriever = HybridRRFRetriever(
            dense, bm25, candidate_k=cfg.get("candidate_k", 12), rrf_k=cfg.get("rrf_k", 60)
        )
    elif method == "hybrid_cross_rerank":
        embedder = GeminiQueryEmbedder(cache_path=cache_path)
        dense = PgVectorDenseRetriever(database_url, embedder, db_strategy="C", dimensions=768)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRRFRetriever(dense, bm25, candidate_k=12, rrf_k=cfg.get("rrf_k", 60))
        reranker = CrossFeatureReranker()
        retriever = RerankedRetriever(hybrid, reranker, candidate_k=cfg.get("candidate_k", 12))
    elif method == "hybrid_rerank":
        embedder = GeminiQueryEmbedder(cache_path=cache_path)
        dense = PgVectorDenseRetriever(database_url, embedder, db_strategy="C", dimensions=768)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRRFRetriever(dense, bm25, candidate_k=12, rrf_k=cfg.get("rrf_k", 60))
        reranker = GeminiReranker(cache_path=cache_path.parent / ".rerank_cache.json")
        retriever = RerankedRetriever(hybrid, reranker, candidate_k=cfg.get("candidate_k", 10))
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

    return retriever, embedder, reranker


def evaluate_config(
    config_name: str,
    cfg: dict[str, Any],
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    split: str,
    database_url: str,
    cache_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [q for q in questions if q.get("split") == split and q.get("answerable", True)]
    if not selected:
        raise ValueError(f"No answerable questions found for split={split!r}")

    retriever, embedder, reranker = build_retriever(
        cfg, chunks, database_url=database_url, cache_path=cache_path
    )
    neighbor_store = CorpusNeighborStore(chunks) if cfg.get("neighbor_window", 0) > 0 else None
    top_k = int(cfg.get("top_k", 3))
    budget = int(cfg.get("budget", 800))

    traces: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for q in selected:
        flt = q.get("filter")
        t0 = time.perf_counter()
        ranked = retriever.search(q["question"], filters=flt, top_k=top_k)

        pool = ranked
        if neighbor_store is not None:
            pool = neighbor_store.expand_results(
                ranked,
                window=int(cfg["neighbor_window"]),
                max_anchors=int(cfg.get("neighbor_anchors", 2)),
                filters=flt,
            )

        context, context_ids = pack_context_budget(pool, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        by_id = {c["id"]: c for c in pool}
        packed_chunks = [by_id[cid] for cid in context_ids if cid in by_id]
        top3 = ranked[:3]
        marked_top3 = [{**c, "hit": chunk_hits_gold(c, q["gold"])} for c in top3]

        hit1 = bool(marked_top3 and marked_top3[0]["hit"])
        hit3 = any(c["hit"] for c in marked_top3)
        ctx_hit = context_hits_gold(q, context, packed_chunks)

        traces.append(
            {
                "query_id": q["id"],
                "split": split,
                "config": config_name,
                "question": q["question"],
                "filter": flt,
                "gold": q["gold"],
                "required_spans": q.get("required_spans"),
                "top3_results": marked_top3,
                "context_ids": context_ids,
                "context_chars": len(context),
                "context": context,
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": hit1,
                "hit3": hit3,
                "context_hit": ctx_hit,
            }
        )

    total_seconds = time.perf_counter() - t_all
    n = len(traces)
    hit1_n = sum(t["hit1"] for t in traces)
    hit3_n = sum(t["hit3"] for t in traces)
    ctx_n = sum(t["context_hit"] for t in traces)

    summary: dict[str, Any] = {
        "config": config_name,
        "description": cfg["description"],
        "split": split,
        "n": n,
        "hit1": f"{hit1_n}/{n}",
        "hit1_rate": round(hit1_n / n, 4),
        "evidence_hit3": f"{hit3_n}/{n}",
        "evidence_hit3_rate": round(hit3_n / n, 4),
        "context_evidence_rate": f"{ctx_n}/{n}",
        "context_evidence_pct": round(ctx_n / n, 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "total_seconds": round(total_seconds, 3),
        "embed_stats": embedder.stats if embedder else None,
        "rerank_stats": getattr(reranker, "stats", None) if reranker else None,
        "service_cost_usd": 0.0,
        "cost_note": "Gemini API Free Tier ($0 billed); token/call counts logged in stats.",
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 retrieval benchmark presets")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks_day6_c.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "benchmark_questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--configs", nargs="+", default=["dense_b800"], choices=list(CONFIG_PRESETS.keys()))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "retrieval")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ROOT / "results" / ".query_embedding_cache.json"
    all_summaries: list[dict[str, Any]] = []

    for name in args.configs:
        summary, traces = evaluate_config(
            name,
            CONFIG_PRESETS[name],
            chunks,
            questions,
            split=args.split,
            database_url=database_url,
            cache_path=cache_path,
        )
        all_summaries.append(summary)
        (args.out_dir / f"{name}_{args.split}_traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.out_dir / f"{name}_{args.split}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))

    (args.out_dir / f"comparison_{args.split}.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
