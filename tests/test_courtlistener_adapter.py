"""Tests for CourtListener adapter (Phase 6 Slice 5b)."""

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

from core.knowledge.adapters import courtlistener  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SEARCH = json.loads(
    (_FIXTURES / "courtlistener_search_miranda.json").read_text(encoding="utf-8")
)


class TestCourtListenerAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_entry_maps_scotus_authority(self) -> None:
        entry = _SEARCH["results"][0]
        row = courtlistener._row_from_entry(entry)
        self.assertEqual(row["_adapter"], "courtlistener")
        self.assertIn("Miranda", row["title"])
        self.assertEqual(row["court_id"], "scotus")
        self.assertGreaterEqual(row["authority_score"], 0.9)
        self.assertTrue(str(row["url"]).startswith("https://www.courtlistener.com/"))

    @patch.object(courtlistener, "fetch_search_results", return_value=_SEARCH)
    def test_search_courtlistener_fixture(self, _mock_fetch) -> None:
        rows = courtlistener.search_courtlistener("Miranda v Arizona", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "courtlistener")

    def test_opinion_snippet_prefers_lead_opinion(self) -> None:
        entry = next(
            result
            for result in _SEARCH["results"]
            if result.get("cluster_id") == 107252
        )
        snippet = courtlistener._opinion_snippet(entry)
        self.assertIn("delivered the opinion of the Court", snippet)

    @patch.object(courtlistener, "fetch_search_results", return_value=_SEARCH)
    def test_search_ranks_landmark_miranda_cluster_first(self, _mock_fetch) -> None:
        rows = courtlistener.search_courtlistener("Miranda v Arizona", max_results=1)
        self.assertEqual(rows[0]["cluster_id"], 107252)
        self.assertIn("delivered the opinion of the Court", rows[0]["snippet"])


if __name__ == "__main__":
    unittest.main()
