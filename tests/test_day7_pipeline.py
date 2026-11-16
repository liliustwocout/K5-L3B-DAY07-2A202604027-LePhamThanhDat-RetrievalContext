"""Unit tests for Day 7 retrieval methods, context packing, and agent adapters (offline mocks)."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_pipeline import MultiToolAgenticRAG, StarterControllerAdapter, pack_recency_aware_context
from retrieval_eval import chunk_hits_gold, context_hits_gold
from retrievers import (
    BM25Retriever,
    CorpusNeighborStore,
    matches_filter,
    pack_context_budget,
    reciprocal_rank_fusion,
)


def make_chunk(cid, text, *, source_id="doc1", page=1, chunk_index=0, year=2024, category="genai"):
    return {
        "id": cid,
        "source_id": source_id,
        "source": f"{source_id}.pdf",
        "page": page,
        "text": text,
        "metadata": {"year": year, "category": category, "chunk_index": chunk_index},
    }


class Day7RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            make_chunk("d1:p1:0", "PW.3.2 Provenance verification is not possible in all cases.", page=1, chunk_index=0),
            make_chunk("d1:p1:1", "However, it is still beneficial to track provenance when unknown.", page=1, chunk_index=1),
            make_chunk("d2:p2:0", "Explainability principles under GDPR Article 13.", source_id="doc2", page=2, chunk_index=0, year=2021, category="explainability"),
        ]

    def test_bm25_and_metadata_filter(self):
        bm25 = BM25Retriever(self.chunks)
        res = bm25.search("PW.3.2 provenance", filters={"category": "genai"}, top_k=3)
        self.assertEqual([r["id"] for r in res], ["d1:p1:0", "d1:p1:1"])
        self.assertTrue(matches_filter(self.chunks[2], {"year": 2021, "category": "explainability"}))
        self.assertFalse(matches_filter(self.chunks[0], {"year": 2021}))

    def test_rrf_combines_ranks_fairly(self):
        dense = [{"id": "d1:p1:1", **self.chunks[1]}, {"id": "d1:p1:0", **self.chunks[0]}]
        lex = [{"id": "d1:p1:0", **self.chunks[0]}, {"id": "d2:p2:0", **self.chunks[2]}]
        fused = reciprocal_rank_fusion([("dense", dense), ("bm25", lex)], rrf_k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["id"], "d1:p1:0")
        self.assertIn("dense", fused[0]["sub_ranks"])
        self.assertIn("bm25", fused[0]["sub_ranks"])

    def test_neighbor_expansion_and_multi_span_context_hit(self):
        store = CorpusNeighborStore(self.chunks)
        expanded = store.expand_results([self.chunks[1]], window=1, max_anchors=1, filters={"category": "genai"})
        self.assertEqual([c["id"] for c in expanded], ["d1:p1:1", "d1:p1:0"])

        q_multi = {
            "gold": [{"source_id": "doc1", "page": 1, "evidence": "Provenance verification ... unknown."}],
            "required_spans": [
                "Provenance verification is not possible in all cases.",
                "However, it is still beneficial to track provenance when unknown.",
            ],
        }
        # Single chunk alone misses multi-span context
        ctx1, ids1 = pack_context_budget([self.chunks[1]], budget=1000)
        self.assertFalse(chunk_hits_gold(self.chunks[1], q_multi["gold"]))
        self.assertFalse(context_hits_gold(q_multi, ctx1, [self.chunks[1]]))

        # Expanded neighbor context satisfies both required spans
        ctx2, ids2 = pack_context_budget(expanded, budget=1000)
        self.assertEqual(set(ids2), {"d1:p1:0", "d1:p1:1"})
        self.assertTrue(context_hits_gold(q_multi, ctx2, expanded))

    def test_recency_aware_context_avoids_starving_later_search(self):
        old_big = make_chunk("old", "A" * 400)
        new_r1 = make_chunk("new_r1", "Exact answer here.")
        new_r2 = make_chunk("new_r2", "Secondary detail " + "B" * 200)
        ctx, ids = pack_recency_aware_context([[old_big], [new_r1, new_r2]], budget=160)
        self.assertEqual(ids, ["new_r1"])
        self.assertIn("Exact answer here.", ctx)


class Day7AgentMockTests(unittest.TestCase):
    def test_starter_controller_parses_search_and_answer(self):
        adapter = StarterControllerAdapter.__new__(StarterControllerAdapter)
        adapter.model_name = "mock"
        adapter.token_stats = {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "llm_seconds": 0.0}

        class FakeModels:
            def __init__(self, payloads):
                self.payloads = iter(payloads)

            def generate_content(self, **kwargs):
                txt = next(self.payloads)
                return SimpleNamespace(
                    text=txt,
                    usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
                )

        adapter.client = SimpleNamespace(
            models=FakeModels([
                '{"action": "search", "query": "PW.3.2"}',
                '{"action": "answer", "answer": "Track provenance [d1:p1:0].", "citations": ["d1:p1:0"]}',
            ])
        )
        self.assertEqual(adapter.decide("Q?", []), {"action": "search", "query": "PW.3.2"})
        self.assertEqual(
            adapter.decide("Q?", [{"queries": ["PW.3.2"], "context": "[d1:p1:0] ..."}]),
            {"action": "answer", "answer": "Track provenance [d1:p1:0].", "citations": ["d1:p1:0"]},
        )

    def test_multi_tool_agent_executes_tools_and_validates_citations(self):
        chunks = [
            make_chunk("c1", "PW.3.2 requires tracking provenance."),
            make_chunk("c2", "Follow-up detail on unknown provenance.", chunk_index=1),
        ]
        bm25 = BM25Retriever(chunks)
        store = CorpusNeighborStore(chunks)

        class FakeHybrid:
            def search(self, query, *, filters=None, top_k=3):
                return bm25.search(query, filters=filters, top_k=top_k)

        responses = iter([
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    function_call=SimpleNamespace(name="search_hybrid", args={"query": "PW.3.2", "top_k": 1}),
                                    text=None,
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=20, candidates_token_count=10),
            ),
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    function_call=SimpleNamespace(name="get_adjacent_chunks", args={"chunk_id": "c1", "window": 1}),
                                    text=None,
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=30, candidates_token_count=10),
            ),
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    function_call=None,
                                    text=json.dumps(
                                        {
                                            "status": "answered",
                                            "answer": "Track provenance [c1] and document unknown [c2].",
                                            "citations": ["c1", "c2"],
                                        }
                                    ),
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=40, candidates_token_count=15),
            ),
        ])

        agent = MultiToolAgenticRAG.__new__(MultiToolAgenticRAG)
        agent.model_name = "mock"
        agent.hybrid_retriever = FakeHybrid()
        agent.lexical_retriever = bm25
        agent.neighbor_store = store
        agent.max_searches = 4
        agent.budget = 2400
        agent._tools = None
        agent.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: next(responses)))

        res = agent.run_case({"id": "q_mock", "split": "dev", "question": "What does PW.3.2 require?", "filter": None})
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["c1", "c2"])
        self.assertEqual(res["num_searches"], 2)
        self.assertEqual([e["tool"] for e in res["events"] if e["event"] == "search"], ["search_hybrid", "get_adjacent_chunks"])


if __name__ == "__main__":
    unittest.main()
