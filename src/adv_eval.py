"""Manual audit & metrics aggregator for Day 7 Advanced Multimodal RAG."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_manual_reviews(
    *,
    benchmark_cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Validate manual pass/fail grades across every case in `split` and compute per-group accuracy."""
    cases = [c for c in benchmark_cases if c["split"] == split]
    trace_by_id = {t["query_id"]: t for t in traces}
    review_by_id = {r["query_id"]: r for r in reviews}

    missing_traces = [c["id"] for c in cases if c["id"] not in trace_by_id]
    if missing_traces:
        raise ValueError(f"Missing traces for {split}: {missing_traces}")
    missing_reviews = [c["id"] for c in cases if c["id"] not in review_by_id]
    if missing_reviews:
        raise ValueError(f"Missing manual reviews for {split}: {missing_reviews}")

    by_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "answered": 0, "abstain": 0, "error": 0}
    )
    rows: list[dict[str, Any]] = []
    total_passed = 0
    total_errors = 0
    total_abstain = 0
    total_answered = 0
    total_searches = 0
    total_visual_inspections = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_elapsed_ms = 0.0

    for c in cases:
        qid = c["id"]
        grp = c["group"]
        tr = trace_by_id[qid]
        rv = review_by_id[qid]
        verdict = str(rv.get("verdict") or "").strip().lower()
        if verdict not in ("pass", "fail"):
            raise ValueError(f"Invalid verdict {verdict!r} for {qid}; expected 'pass' or 'fail'")

        status = tr.get("status")
        if status == "error" and verdict == "pass":
            raise ValueError(f"Error trace {qid} cannot be graded 'pass'")

        stats = by_group[grp]
        stats["total"] += 1
        if verdict == "pass":
            stats["passed"] += 1
            total_passed += 1
        else:
            stats["failed"] += 1

        if status == "answered":
            stats["answered"] += 1
            total_answered += 1
        elif status == "abstain":
            stats["abstain"] += 1
            total_abstain += 1
        else:
            stats["error"] += 1
            total_errors += 1

        total_searches += int(tr.get("num_searches") or 0)
        total_visual_inspections += int(tr.get("visual_inspections") or 0)
        tu = tr.get("token_usage") or {}
        total_prompt_tokens += int(tu.get("prompt_tokens") or 0)
        total_output_tokens += int(tu.get("output_tokens") or 0)
        total_elapsed_ms += float(tr.get("elapsed_ms") or 0.0)

        rows.append(
            {
                "query_id": qid,
                "group": grp,
                "status": status,
                "verdict": verdict,
                "citations": tr.get("citations", []),
                "visual_inspections": int(tr.get("visual_inspections") or 0),
                "reason": rv.get("reason", ""),
            }
        )

    total = len(cases)
    group_summary = {
        g: {
            **s,
            "accuracy": (s["passed"] / s["total"]) if s["total"] else 0.0,
        }
        for g, s in by_group.items()
    }
    return {
        "split": split,
        "total": total,
        "passed": total_passed,
        "failed": total - total_passed,
        "accuracy": (total_passed / total) if total else 0.0,
        "accuracy_pct": f"{(100.0 * total_passed / total):.2f}%" if total else "0.00%",
        "status_counts": {"answered": total_answered, "abstain": total_abstain, "error": total_errors},
        "avg_tool_calls": round(total_searches / total, 2) if total else 0.0,
        "avg_visual_inspections": round(total_visual_inspections / total, 2) if total else 0.0,
        "avg_elapsed_s": round((total_elapsed_ms / 1000.0) / total, 2) if total else 0.0,
        "token_usage": {"prompt_tokens": total_prompt_tokens, "output_tokens": total_output_tokens},
        "by_group": group_summary,
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate manual review grades for Day 7 Advanced")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark_cases = json.loads(args.benchmark.read_text(encoding="utf-8"))
    traces = json.loads(args.traces.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))

    summary = summarize_manual_reviews(
        benchmark_cases=benchmark_cases,
        traces=traces,
        reviews=reviews,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
