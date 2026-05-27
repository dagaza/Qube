"""Hybrid merge of dual-query retrieval results."""
from __future__ import annotations

import unittest

from core.dual_query_retrieval import merge_memory_search_results, merge_rag_search_results


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


if __name__ == "__main__":
    unittest.main()
