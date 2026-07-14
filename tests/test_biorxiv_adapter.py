"""Tests for bioRxiv adapter (Phase 6c-1)."""

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

from core.knowledge.adapters import biorxiv  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_CRISPR = json.loads((_FIXTURES / "biorxiv_search_crispr.json").read_text(encoding="utf-8"))


class TestBiorxivAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_entry_maps_metadata(self) -> None:
        entry = _CRISPR["results"][0]
        row = biorxiv._row_from_entry(entry)
        self.assertEqual(row["_adapter"], "biorxiv")
        self.assertIn("CRISPR", row["title"])
        self.assertTrue(row["preprint"])
        self.assertEqual(row["biorxiv_id"], "2023.01.15.524102")

    @patch.object(biorxiv, "fetch_search_results", return_value=_CRISPR)
    def test_search_biorxiv_fixture(self, _mock_fetch) -> None:
        rows = biorxiv.search_biorxiv(
            "CRISPR Cas9 gene editing off-target effects",
            max_results=2,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "biorxiv")

    def test_live_search_returns_empty_without_fixtures_or_network(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)
        with patch.object(biorxiv, "fetch_search_results", return_value={"results": []}):
            rows = biorxiv.search_biorxiv("microbiome metagenomics")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
