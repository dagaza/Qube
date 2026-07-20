"""Tests for help corpus isolation from unscoped library RAG."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = MagicMock()
if "pyarrow" not in sys.modules:
    sys.modules["pyarrow"] = MagicMock()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.rag_tool import rag_search  # noqa: E402


class HelpCorpusIsolationTests(unittest.TestCase):
    @patch("mcp.rag_tool.logger")
    def test_unscoped_search_excludes_help_docs(self, _log) -> None:
        store = MagicMock()
        table = MagicMock()
        store.table = table
        rows = [
            {
                "source": "qube/documentation/features/library.md",
                "text": "Library help documentation",
                "chunk_id": 0,
                "_distance": 0.05,
            },
            {
                "source": "notes/meeting.pdf",
                "text": "User meeting notes about project alpha",
                "chunk_id": 0,
                "_distance": 0.15,
            },
        ]

        def _search_query(*_args, **_kwargs):
            query = MagicMock()
            query.where.return_value.limit.return_value.to_list.return_value = rows
            query.limit.return_value.to_list.return_value = rows
            return query

        table.search.side_effect = _search_query

        result = rag_search(
            "summarize my meeting notes",
            np.zeros(768),
            store,
            top_k=3,
        )
        filenames = [src["filename"] for src in result.get("sources") or []]
        self.assertNotIn("qube/documentation/features/library.md", filenames)
        self.assertIn("notes/meeting.pdf", filenames)

    @patch("mcp.rag_tool.logger")
    def test_help_prefix_search_keeps_help_docs(self, _log) -> None:
        store = MagicMock()
        table = MagicMock()
        store.table = table
        rows = [
            {
                "source": "qube/documentation/features/settings/ai-models.md",
                "text": "GPU layers in AI and Models settings",
                "chunk_id": 0,
                "_distance": 0.05,
            }
        ]

        def _search_query(*_args, **_kwargs):
            query = MagicMock()
            query.where.return_value.limit.return_value.to_list.return_value = rows
            query.limit.return_value.to_list.return_value = rows
            return query

        table.search.side_effect = _search_query

        result = rag_search(
            "gpu layers",
            np.zeros(768),
            store,
            top_k=3,
            source_prefix_filter="qube/documentation/",
        )
        self.assertEqual(len(result.get("sources") or []), 1)


if __name__ == "__main__":
    unittest.main()
