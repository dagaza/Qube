"""Hybrid merge of dual-query retrieval results."""
from __future__ import annotations

import unittest

from core.dual_query_retrieval import (
    merge_memory_search_results,
    merge_rag_search_results,
    merge_web_search_results,
)


class TestDualQueryRetrieval(unittest.TestCase):
    def test_merge_memory_dedupes_by_memory_id(self) -> None:
        primary = {
            "memory_context": "- fact a",
            "memory_sources": [
                {"id": 1, "memory_id": "m1", "content": "fact a", "category": "preference"},
            ],
        }
        aux = {
            "memory_context": "- fact b",
            "memory_sources": [
                {"id": 1, "memory_id": "m1", "content": "fact a dup", "category": "preference"},
                {"id": 2, "memory_id": "m2", "content": "fact b", "category": "knowledge"},
            ],
        }
        merged = merge_memory_search_results(primary, aux)
        ids = {s["memory_id"] for s in merged["memory_sources"]}
        self.assertEqual(ids, {"m1", "m2"})
        self.assertEqual(len(merged["memory_sources"]), 2)
        # Primary row wins on duplicate key (preserves primary content).
        m1 = next(s for s in merged["memory_sources"] if s["memory_id"] == "m1")
        self.assertEqual(m1["content"], "fact a")

    def test_merge_memory_rrf_promotes_auxiliary_top_hit(self) -> None:
        """Auxiliary rank-0 hit outranks weak primary-only tail rows."""
        primary = {
            "memory_context": "",
            "memory_sources": [
                {"memory_id": "m1", "content": "weak primary a"},
                {"memory_id": "m2", "content": "weak primary b"},
                {"memory_id": "m3", "content": "weak primary c"},
            ],
        }
        aux = {
            "memory_context": "",
            "memory_sources": [
                {"memory_id": "m4", "content": "strong expanded hit"},
            ],
        }
        merged = merge_memory_search_results(primary, aux)
        order = [s["memory_id"] for s in merged["memory_sources"]]
        self.assertLess(order.index("m4"), order.index("m3"))
        self.assertLess(order.index("m4"), order.index("m2"))

    def test_merge_memory_rrf_boosts_dual_list_hits(self) -> None:
        """A row present in both lists ranks above primary-only rows."""
        primary = {
            "memory_context": "",
            "memory_sources": [
                {"memory_id": "m1", "content": "primary only"},
                {"memory_id": "m_shared", "content": "shared from primary"},
            ],
        }
        aux = {
            "memory_context": "",
            "memory_sources": [
                {"memory_id": "m_shared", "content": "shared from aux"},
                {"memory_id": "m2", "content": "aux only"},
            ],
        }
        merged = merge_memory_search_results(primary, aux)
        order = [s["memory_id"] for s in merged["memory_sources"]]
        self.assertEqual(order[0], "m_shared")

    def test_merge_rag_by_chunk_id(self) -> None:
        primary = {
            "llm_context": "--- SOURCE 1: a.pdf ---\nalpha",
            "sources": [{"id": 1, "chunk_id": "a.pdf::0", "content": "alpha", "filename": "a.pdf"}],
        }
        aux = {
            "llm_context": "--- SOURCE 1: b.pdf ---\nbeta",
            "sources": [{"id": 1, "chunk_id": "b.pdf::0", "content": "beta", "filename": "b.pdf"}],
        }
        merged = merge_rag_search_results(primary, aux)
        self.assertEqual(len(merged["sources"]), 2)

    def test_merge_rag_rrf_ordering(self) -> None:
        primary = {
            "llm_context": "",
            "sources": [
                {"chunk_id": "a::0", "content": "a", "filename": "a.pdf"},
                {"chunk_id": "b::0", "content": "b", "filename": "b.pdf"},
            ],
        }
        aux = {
            "llm_context": "",
            "sources": [
                {"chunk_id": "c::0", "content": "c", "filename": "c.pdf"},
            ],
        }
        merged = merge_rag_search_results(primary, aux)
        order = [s["chunk_id"] for s in merged["sources"]]
        self.assertLess(order.index("c::0"), order.index("b::0"))

    def test_merge_web_search_results_dedupes_by_url(self) -> None:
        primary = [
            {"title": "A", "snippet": "alpha", "url": "https://example.com/a"},
            {"title": "B", "snippet": "beta", "url": "https://example.com/b"},
        ]
        aux = [
            {"title": "A dup", "snippet": "alpha dup", "url": "https://example.com/a"},
            {"title": "C", "snippet": "gamma", "url": "https://example.com/c"},
        ]
        merged = merge_web_search_results(primary, aux, max_results=5)
        urls = [r["url"] for r in merged]
        self.assertEqual(len(urls), 3)
        self.assertEqual(len(set(urls)), 3)
        self.assertEqual(merged[0]["url"], "https://example.com/a")

    def test_merge_web_search_results_rrf_promotes_aux_top(self) -> None:
        primary = [
            {"title": "Old", "snippet": "old1"},
            {"title": "Old2", "snippet": "old2"},
            {"title": "Old3", "snippet": "old3"},
        ]
        aux = [
            {"title": "Fresh", "snippet": "best expanded match", "url": "https://example.com/fresh"},
        ]
        merged = merge_web_search_results(primary, aux, max_results=5)
        order = [r["title"] for r in merged]
        self.assertLess(order.index("Fresh"), order.index("Old2"))


if __name__ == "__main__":
    unittest.main()
