"""Offline unit tests for Day 7 Advanced Multimodal RAG pipeline and benchmark hygiene."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adv_agent  # noqa: E402
import adv_eval  # noqa: E402
from adv_agent import AdvBM25Retriever, AdvHybridRetriever, MultiModalAdvAgent  # noqa: E402
from retrievers import _is_retryable_api_error  # noqa: E402


def _make_sample_chunks() -> list[dict]:
    return [
        {
            "id": "arc2025:p007:fig:00",
            "source_id": "arctic_report_card_2025",
            "source": "Arctic Report Card 2025",
            "page": 7,
            "modality": "visual",
            "visual_type": "diagram",
            "section": "Overview",
            "crop_path": "crops/p007_fig_00.png",
            "text": "[VISUAL:DIAGRAM] Vital Signs wheel showing 8 Arctic indicators including Snow Cover.",
            "metadata": {"bbox": [10.0, 20.0, 400.0, 300.0]},
        },
        {
            "id": "arc2025:p046:table:00",
            "source_id": "arctic_report_card_2025",
            "source": "Arctic Report Card 2025",
            "page": 46,
            "modality": "table",
            "visual_type": None,
            "section": "Sea Ice",
            "crop_path": "crops/p046_table_00.png",
            "text": "Table 1 March September sea ice extent 14.12 4.75 rank 1",
            "metadata": {"bbox": [20.0, 40.0, 500.0, 260.0]},
        },
        {
            "id": "arc2025:p081:text:002",
            "source_id": "arctic_report_card_2025",
            "source": "Arctic Report Card 2025",
            "page": 81,
            "modality": "text",
            "visual_type": None,
            "section": "Glaciers",
            "crop_path": None,
            "text": "The Arctic contains 60% of the world mountain glaciers and ice caps by area.",
            "metadata": {"bbox": [30.0, 50.0, 500.0, 180.0]},
        },
    ]


class _FakeDense:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits

    def search(self, query: str, *, modality: str | None = None, page: int | None = None, top_k: int = 5):
        out = [h for h in self.hits if (modality is None or h["modality"] == modality) and (page is None or h["page"] == page)]
        return [{**h, "rank": i + 1, "score": 0.9 - i * 0.1, "strategy": "adv_dense"} for i, h in enumerate(out[:top_k])]


def _fake_resp(parts: list[object]) -> SimpleNamespace:
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


def _part_fn(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None)


def _part_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(function_call=None, text=text)


class TestAdvPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self.tmp.name)
        (self.tmp_root / "crops").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=(30, 120, 200)).save(self.tmp_root / "crops" / "p007_fig_00.png")

        self.pdf_path = self.tmp_root / "sample.pdf"
        doc = pymupdf.open()
        for page_idx in range(1, 85):
            page = doc.new_page(width=595, height=842)
            if page_idx in (7, 46, 81):
                page.insert_text((72, 72), f"Synthetic PDF page {page_idx}")
        doc.save(self.pdf_path)
        doc.close()

        self.chunks = _make_sample_chunks()
        self.lexical = AdvBM25Retriever(self.chunks)
        self.hybrid = AdvHybridRetriever(_FakeDense(self.chunks), self.lexical, candidate_k=5, rrf_k=60)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bm25_modality_and_page_filters(self) -> None:
        hits = self.lexical.search("Arctic mountain glaciers 60%", modality="text", top_k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "arc2025:p081:text:002")

        miss = self.lexical.search("penguin Antarctic treaty", top_k=3)
        self.assertEqual(miss, [])

        by_page = self.hybrid.search("sea ice extent", modality="table", page=46, top_k=3)
        self.assertEqual(by_page[0]["id"], "arc2025:p046:table:00")

    def test_multimodal_agent_inspects_crop_and_page_image(self) -> None:
        responses = [
            # Turn 1: tries answering without tools -> rejected by loop
            _fake_resp([_part_text('{"status":"answered","answer":"50%","citations":["arc2025:p007:fig:00"]}')]),
            # Turn 2: calls search_multimodal
            _fake_resp([_part_fn("search_multimodal", {"query": "Vital Signs Snow Cover", "top_k": 3})]),
            # Turn 3: calls inspect_visual_region on existing crop + inspect_page_image on page 46
            _fake_resp(
                [
                    _part_fn("inspect_visual_region", {"chunk_id": "arc2025:p007:fig:00"}),
                    _part_fn("inspect_page_image", {"page": 46}),
                ]
            ),
            # Turn 4: returns valid grounded JSON
            _fake_resp(
                [
                    _part_text(
                        json.dumps(
                            {
                                "status": "answered",
                                "answer": "June snow cover extent reduced by 50% [arc2025:p007:fig:00].",
                                "citations": ["arc2025:p007:fig:00"],
                            }
                        )
                    )
                ]
            ),
        ]
        captured_contents: list[list] = []

        def fake_gen(_client, *, model, contents, config, token_stats):
            captured_contents.append(list(contents))
            return responses.pop(0)

        with patch.object(adv_agent, "ROOT", self.tmp_root), patch.object(adv_agent, "_generate_with_retry", side_effect=fake_gen):
            agent = MultiModalAdvAgent(
                model_name="gemini-3.5-flash-lite",
                api_key="test-key",
                chunks=self.chunks,
                pdf_path=self.pdf_path,
                hybrid=self.hybrid,
                lexical=self.lexical,
                enable_vision_tools=True,
                max_steps=6,
            )
            trace = agent.run_case(
                {
                    "id": "t_img",
                    "split": "dev",
                    "group": "image_diagram",
                    "question": "What Snow Cover trend is on the Vital Signs diagram?",
                }
            )

        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["num_searches"], 3)
        self.assertEqual(trace["visual_inspections"], 2)
        self.assertEqual(trace["citations"], ["arc2025:p007:fig:00"])
        self.assertEqual(trace["citation_evidence"][0]["crop_path"], "crops/p007_fig_00.png")

        # Verify PNG image bytes were attached to conversation before turn 4
        last_user_parts = captured_contents[-1][-1].parts
        png_parts = [p for p in last_user_parts if getattr(p, "inline_data", None) and p.inline_data.mime_type == "image/png"]
        self.assertEqual(len(png_parts), 2)

    def test_hallucinated_citation_forces_abstain(self) -> None:
        responses = [
            _fake_resp([_part_fn("search_lexical", {"query": "glaciers", "top_k": 2})]),
            _fake_resp(
                [
                    _part_text(
                        json.dumps(
                            {
                                "status": "answered",
                                "answer": "Fake claim [arc2025:p999:text:999].",
                                "citations": ["arc2025:p999:text:999"],
                            }
                        )
                    )
                ]
            ),
            _fake_resp(
                [
                    _part_text(
                        json.dumps(
                            {
                                "status": "answered",
                                "answer": "Still fake [arc2025:p999:text:999].",
                                "citations": ["arc2025:p999:text:999"],
                            }
                        )
                    )
                ]
            ),
            _fake_resp(
                [
                    _part_text(
                        json.dumps(
                            {
                                "status": "answered",
                                "answer": "Third fake [arc2025:p999:text:999].",
                                "citations": ["arc2025:p999:text:999"],
                            }
                        )
                    )
                ]
            ),
        ]
        with patch.object(adv_agent, "_generate_with_retry", side_effect=responses):
            agent = MultiModalAdvAgent(
                model_name="gemini-3.5-flash-lite",
                api_key="test-key",
                chunks=self.chunks,
                pdf_path=self.pdf_path,
                hybrid=self.hybrid,
                lexical=self.lexical,
                enable_vision_tools=False,
            )
            try:
                trace = agent.run_case(
                    {"id": "t_bad", "split": "dev", "group": "text", "question": "Glacier percentage?"}
                )
            finally:
                agent.pdf_doc.close()
        self.assertEqual(trace["status"], "abstain")
        self.assertEqual(trace["abstain_reason"], "invalid_or_missing_citations")
        self.assertEqual(trace["citations"], [])

    def test_comma_joined_multihop_citations_accepted(self) -> None:
        responses = [
            _fake_resp([_part_fn("search_lexical", {"query": "Arctic glaciers sea ice", "top_k": 3})]),
            _fake_resp(
                [
                    _part_text(
                        json.dumps(
                            {
                                "status": "answered",
                                "answer": "60% glaciers and sea ice [arc2025:p081:text:002, arc2025:p046:table:00].",
                                "citations": ["arc2025:p081:text:002", "arc2025:p046:table:00"],
                            }
                        )
                    )
                ]
            ),
        ]
        with patch.object(adv_agent, "_generate_with_retry", side_effect=responses):
            agent = MultiModalAdvAgent(
                model_name="gemini-3.5-flash-lite",
                api_key="test-key",
                chunks=self.chunks,
                pdf_path=self.pdf_path,
                hybrid=self.hybrid,
                lexical=self.lexical,
                enable_vision_tools=False,
            )
            try:
                trace = agent.run_case(
                    {"id": "t_mh", "split": "dev", "group": "multi_hop", "question": "Compare both"}
                )
            finally:
                agent.pdf_doc.close()
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["arc2025:p081:text:002", "arc2025:p046:table:00"])

    def test_benchmark_quotas_and_disjoint_splits(self) -> None:
        bm_path = ROOT / "data" / "adv_benchmark.json"
        self.assertTrue(bm_path.exists(), "data/adv_benchmark.json must exist")
        cases = json.loads(bm_path.read_text(encoding="utf-8"))
        dev = [c for c in cases if c["split"] == "dev"]
        test = [c for c in cases if c["split"] == "test"]
        self.assertGreaterEqual(sum(1 for c in dev if c["answerable"]), 15)
        self.assertGreaterEqual(sum(1 for c in dev if not c["answerable"]), 3)
        self.assertGreaterEqual(len(test), 30)

        from collections import Counter

        counts = Counter(c["group"] for c in test)
        self.assertGreaterEqual(counts["table"], 6)
        self.assertGreaterEqual(counts["image_diagram"], 5)
        self.assertGreaterEqual(counts["chart"], 5)
        self.assertGreaterEqual(counts["multi_hop"], 4)
        self.assertGreaterEqual(counts["text"], 4)
        self.assertGreaterEqual(counts["unanswerable"], 6)

        dev_pages = {p for c in dev for p in c["expected_pages"]}
        test_pages = {p for c in test for p in c["expected_pages"]}
        self.assertEqual(dev_pages & test_pages, set())
        self.assertEqual(len({c["id"] for c in cases}), len(cases))

    def test_adv_eval_rejects_passed_error_and_computes_exact_rates(self) -> None:
        bm = [
            {"id": "q1", "split": "test", "group": "table"},
            {"id": "q2", "split": "test", "group": "chart"},
        ]
        traces = [
            {"query_id": "q1", "status": "answered", "num_searches": 2, "visual_inspections": 1, "citations": ["c1"]},
            {"query_id": "q2", "status": "error", "num_searches": 0, "visual_inspections": 0, "citations": []},
        ]
        with self.assertRaises(ValueError):
            adv_eval.summarize_manual_reviews(
                benchmark_cases=bm,
                traces=traces,
                reviews=[{"query_id": "q1", "verdict": "pass"}, {"query_id": "q2", "verdict": "pass"}],
                split="test",
            )
        summary = adv_eval.summarize_manual_reviews(
            benchmark_cases=bm,
            traces=traces,
            reviews=[
                {"query_id": "q1", "verdict": "pass", "reason": "exact"},
                {"query_id": "q2", "verdict": "fail", "reason": "api error"},
            ],
            split="test",
        )
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["accuracy_pct"], "50.00%")
        self.assertEqual(summary["status_counts"]["error"], 1)

    def test_empty_json_response_recovers_and_protocol_retry(self) -> None:
        class RemoteProtocolError(Exception):
            pass

        self.assertTrue(_is_retryable_api_error(RemoteProtocolError("Server disconnected without sending a response.")))

        responses = [
            _fake_resp([_part_fn("search_lexical", {"query": "glaciers", "top_k": 2})]),
            _fake_resp([]),
            _fake_resp(
                [
                    _part_text(
                        json.dumps(
                            {
                                "status": "answered",
                                "answer": "The Arctic contains 60% [arc2025:p081:text:002].",
                                "citations": ["arc2025:p081:text:002"],
                            }
                        )
                    )
                ]
            ),
        ]
        with patch.object(adv_agent, "_generate_with_retry", side_effect=responses):
            agent = MultiModalAdvAgent(
                model_name="gemini-3.1-flash-lite",
                api_key="test-key",
                chunks=self.chunks,
                pdf_path=self.pdf_path,
                hybrid=self.hybrid,
                lexical=self.lexical,
                enable_vision_tools=False,
            )
            try:
                trace = agent.run_case({"id": "t_json", "split": "dev", "group": "text", "question": "Glaciers?"})
            finally:
                agent.pdf_doc.close()
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["arc2025:p081:text:002"])


if __name__ == "__main__":
    unittest.main()
