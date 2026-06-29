"""Tests for DBLP adapter (Phase 6 Slice 6b)."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters import dblp  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SEARCH = json.loads(
    (_FIXTURES / "dblp_search_transformer.json").read_text(encoding="utf-8")
)


class TestDblpAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_hit_maps_authors(self) -> None:
        hit = _SEARCH["result"]["hits"]["hit"][0]
        row = dblp._row_from_hit(hit)
        assert row is not None
        self.assertEqual(row["_adapter"], "dblp")
        self.assertIn("Attention Is All You Need", row["title"])
        self.assertGreaterEqual(len(row["authors"]), 2)

    @patch.object(dblp, "fetch_search_results", return_value=_SEARCH)
    def test_search_dblp_fixture(self, _mock_fetch) -> None:
        rows = dblp.search_dblp("transformer attention mechanism", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "dblp")


if __name__ == "__main__":
    unittest.main()
