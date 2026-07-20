"""Tests for scoped RAG search (source_filter)."""

import sys
import unittest
from unittest.mock import MagicMock, patch

# rag_tool imports lancedb via rag.store — mock when unavailable in CI/sandbox.
if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = MagicMock()
if "pyarrow" not in sys.modules:
    sys.modules["pyarrow"] = MagicMock()

import numpy as np

from mcp.rag_tool import _filter_results_by_source, rag_search


class TestRagSourceFilter(unittest.TestCase):
    def test_filter_results_by_source(self):
        rows = [
            {"source": "a.pdf", "text": "one"},
            {"source": "b.pdf", "text": "two"},
        ]
        out = _filter_results_by_source(rows, "a.pdf")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["source"], "a.pdf")

    def test_filter_results_by_source_prefix(self):
        from mcp.rag_tool import _filter_results_by_source_prefix

        rows = [
            {"source": "qube/documentation/help.md", "text": "help"},
            {"source": "notes.pdf", "text": "user"},
        ]
        out = _filter_results_by_source_prefix(rows, "qube/documentation/")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["source"], "qube/documentation/help.md")

    @patch("mcp.rag_tool.logger")
    def test_scoped_empty_falls_back_to_reconstruct(self, _log):
        store = MagicMock()
        table = MagicMock()
        store.table = table
        table.search.return_value.where.return_value.limit.return_value.to_list.return_value = []
        store.reconstruct_document.return_value = "Full document body text."

        result = rag_search(
            "summary",
            np.zeros(768),
            store,
            top_k=3,
            source_filter="scoped.pdf",
        )
        self.assertIn("scoped.pdf", result["llm_context"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "scoped.pdf")
        store.reconstruct_document.assert_called_once_with("scoped.pdf")


if __name__ == "__main__":
    unittest.main()
