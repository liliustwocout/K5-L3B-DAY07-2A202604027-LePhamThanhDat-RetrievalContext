"""Build Day 7 benchmark dataset (dev/test, answerable + verified unanswerable cases).

Never leaks gold answers or evidence quotes into `agent_cases.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

UNANSWERABLE_CASES: list[dict[str, Any]] = [
    {
        "id": "u_dev_01",
        "split": "dev",
        "type": "unanswerable_near_topic",
        "question": (
            "Under the GDPR mentions in the NIST explainability publication, what administrative "
            "fine percentages or maximum Euro penalties apply for violating GDPR Article 22 "
            "automated decision-making rules?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NIST IR 8312 briefly cites GDPR Article 13 in explainability context (pp. 8, 16, 30, 42), "
            "but GDPR Article 22, administrative fine percentages, and Euro penalties do not appear anywhere in the corpus."
        ),
        "verification_method": (
            "Full-corpus regex scan across all 159 physical pages of NIST.IR.8312.pdf, NIST.SP.1270.pdf, "
            "and NIST.SP.800-218A.pdf for 'Article 22', 'penalty/penalties', 'euro/million' returned 0 matches."
        ),
        "filter": {"category": "explainability"},
    },
    {
        "id": "u_dev_02",
        "split": "dev",
        "type": "unanswerable_near_topic",
        "question": (
            "According to the secure generative AI development practices profile, what specific "
            "KL-divergence penalty coefficient or PPO clipping epsilon is recommended during RLHF alignment?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NIST SP 800-218A discusses aligning and fine-tuning at a high-level SSDF practice level (e.g., PW.3.2), "
            "but specifies no RLHF algorithm hyperparameters such as KL penalty coefficients or PPO clipping epsilon."
        ),
        "verification_method": (
            "Full-corpus regex scan across all 159 physical pages for 'RLHF', 'PPO', 'KL', 'clipping', "
            "and 'epsilon' returned 0 matches."
        ),
        "filter": {"category": "genai_security"},
    },
    {
        "id": "u_dev_03",
        "split": "dev",
        "type": "unanswerable_near_topic",
        "question": (
            "How does the EU AI Act Article 52 transparency obligation or conformity assessment procedure "
            "map to the four explainable AI principles in the NIST documents?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The NIST documents cover explainability principles and transparency (IR 8312, SP 1270) "
            "but never mention or map to the EU AI Act, Article 52, or conformity assessment procedures."
        ),
        "verification_method": (
            "Full-corpus regex scan across all 159 physical pages for 'EU AI Act', 'Article 52', "
            "and 'conformity assessment' returned 0 matches."
        ),
        "filter": None,
    },
    {
        "id": "u_test_01",
        "split": "test",
        "type": "unanswerable_near_topic",
        "question": (
            "Under task RV.2.2 of the generative AI secure development profile, within how many "
            "calendar days must an organization patch or disclose newly discovered critical AI model vulnerabilities?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NIST SP 800-218A RV.2.2 recommends considering time and rebuilding expenses in risk responses, "
            "but gives no numeric patch or disclosure SLA in calendar days."
        ),
        "verification_method": (
            "Full-corpus regex scan across all 159 physical pages for '\\b\\d+\\s+(days?|hours?|months?)\\b' "
            "and 'calendar days' returned 0 matches."
        ),
        "filter": {"year": 2024},
    },
    {
        "id": "u_test_02",
        "split": "test",
        "type": "unanswerable_near_topic",
        "question": (
            "In SP 800-218A, how does NIST specify implementing cryptographic watermarking or C2PA "
            "provenance metadata tags for generated images and audio?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NIST SP 800-218A addresses training/fine-tuning data provenance under PW.3.2, "
            "but contains no guidance on cryptographic watermarking or C2PA tags for generated media."
        ),
        "verification_method": (
            "Full-corpus regex scan across all 159 physical pages for 'watermark' and 'C2PA' returned 0 matches."
        ),
        "filter": None,
    },
    {
        "id": "u_test_03",
        "split": "test",
        "type": "unanswerable_near_topic",
        "question": (
            "How does ISO/IEC 42001 Clause 9.1 map to the AI bias and explainability governance controls "
            "across the three NIST publications?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The three NIST publications discuss AI governance, bias, and explainability, "
            "but make no reference to ISO/IEC 42001 or Clause 9.1."
        ),
        "verification_method": (
            "Full-corpus regex scan across all 159 physical pages for '42001' and 'Clause 9' returned 0 matches."
        ),
        "filter": None,
    },
]

# q22 spans two adjacent chunks in Day 6 Strategy C (p023:para:000 and p023:para:001).
MULTI_SPAN_OVERRIDES: dict[str, list[str]] = {
    "q22": [
        "Provenance verification is not possible in all cases because provenance is not always known.",
        "However, it is still beneficial for security purposes to track and verify provenance whenever possible, and to track when provenance is unknown.",
    ]
}


def build_benchmark(day6_questions_path: Path, out_questions: Path, out_cases: Path) -> None:
    payload = json.loads(day6_questions_path.read_text(encoding="utf-8"))
    raw_questions = payload["questions"] if isinstance(payload, dict) else payload

    benchmark_questions: list[dict[str, Any]] = []
    agent_cases: list[dict[str, Any]] = []

    for q in raw_questions:
        gold_entry = {
            "source_id": q["gold_source_id"],
            "page": int(q["gold_page"]),
            "evidence": q["evidence_quote"],
        }
        entry: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "type": q.get("type", "direct"),
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": [gold_entry],
            "filter": q.get("filter"),
        }
        if q["id"] in MULTI_SPAN_OVERRIDES:
            entry["required_spans"] = MULTI_SPAN_OVERRIDES[q["id"]]
            entry["multi_chunk_note"] = (
                "Full evidence quote spans adjacent chunks p023:para:000 and p023:para:001 in Strategy C."
            )
        benchmark_questions.append(entry)
        agent_cases.append(
            {
                "id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "answerable": True,
                "expected_behavior": "search_and_answer",
                "filter": q.get("filter"),
            }
        )

    for u in UNANSWERABLE_CASES:
        benchmark_questions.append(u)
        agent_cases.append(
            {
                "id": u["id"],
                "split": u["split"],
                "question": u["question"],
                "answerable": False,
                "expected_behavior": u["expected_behavior"],
                "absence_reason": u["absence_reason"],
                "filter": u.get("filter"),
            }
        )

    out_questions.parent.mkdir(parents=True, exist_ok=True)
    out_questions.write_text(
        json.dumps(benchmark_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_cases.write_text(
        json.dumps(agent_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dev_ans = sum(1 for q in benchmark_questions if q["split"] == "dev" and q["answerable"])
    dev_unans = sum(1 for q in benchmark_questions if q["split"] == "dev" and not q["answerable"])
    test_ans = sum(1 for q in benchmark_questions if q["split"] == "test" and q["answerable"])
    test_unans = sum(1 for q in benchmark_questions if q["split"] == "test" and not q["answerable"])
    print(
        f"Built benchmark: dev={dev_ans} answerable + {dev_unans} unanswerable ({dev_ans + dev_unans}); "
        f"test={test_ans} answerable + {test_unans} unanswerable ({test_ans + test_unans})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Day 7 benchmark_questions.json and agent_cases.json")
    parser.add_argument(
        "--day6-questions",
        type=Path,
        default=ROOT.parent / "Day06" / "benchmark" / "questions.json",
    )
    parser.add_argument("--out-questions", type=Path, default=ROOT / "data" / "benchmark_questions.json")
    parser.add_argument("--out-cases", type=Path, default=ROOT / "data" / "agent_cases.json")
    args = parser.parse_args()
    build_benchmark(args.day6_questions, args.out_questions, args.out_cases)


if __name__ == "__main__":
    main()
