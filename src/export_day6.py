"""Export Day 6 chunks from PostgreSQL (pgvector) and questions JSON into Day 7 format."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from convert_day6 import convert_chunk, convert_question  # noqa: E402


def export_chunks_from_db(database_url: str, strategy: str) -> list[dict]:
    """Read stored chunks for a strategy from Day 6 PostgreSQL table `rag_chunks`."""
    with psycopg.connect(database_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, source, page, text, metadata
                FROM rag_chunks
                WHERE strategy = %s
                ORDER BY document_id, page, chunk_id
                """,
                (strategy,),
            )
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"No rows found in rag_chunks for strategy={strategy!r}")

    return [
        {
            "chunk_id": row[0],
            "document_id": row[1],
            "source": row[2],
            "page": int(row[3]),
            "text": row[4],
            "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5]),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Day 6 Postgres chunks & questions to Day 7 JSON")
    parser.add_argument("--strategy", default="C", help="Day 6 strategy key in rag_chunks (default: C)")
    parser.add_argument(
        "--day6-questions",
        type=Path,
        default=ROOT.parent / "Day06" / "benchmark" / "questions.json",
        help="Path to Day 6 benchmark questions.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "starter" / "data" / "raw_day6",
        help="Directory for raw Day 6 JSON arrays",
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=ROOT / "starter" / "data" / "converted",
        help="Directory for converted Day 7 JSON arrays",
    )
    parser.add_argument(
        "--repo-chunks",
        type=Path,
        default=ROOT / "data" / "chunks_day6_c.json",
        help="Optional tracked copy of converted chunks in data/",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing in .env")

    raw_chunks = export_chunks_from_db(database_url, args.strategy)
    q_payload = json.loads(args.day6_questions.read_text(encoding="utf-8"))
    raw_questions = q_payload["questions"] if isinstance(q_payload, dict) else q_payload
    if not isinstance(raw_questions, list):
        raise ValueError("Day 6 questions payload must be a list or contain a 'questions' list")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    (args.raw_dir / "chunks.json").write_text(
        json.dumps(raw_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.raw_dir / "questions.json").write_text(
        json.dumps(raw_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = [convert_question(q) for q in raw_questions]

    args.converted_dir.mkdir(parents=True, exist_ok=True)
    (args.converted_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.converted_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.repo_chunks:
        args.repo_chunks.parent.mkdir(parents=True, exist_ok=True)
        args.repo_chunks.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"Exported strategy={args.strategy}: {len(chunks)} chunks, {len(questions)} questions -> "
        f"{args.converted_dir} (and {args.repo_chunks})"
    )


if __name__ == "__main__":
    main()
