"""Tests for INSPIRE-HEP adapter (Phase 6c-5)."""

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

from core.knowledge.adapters import inspire_hep  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SEARCH = json.loads(
    (_FIXTURES / "inspire_hep_search_ligo.json").read_text(encoding="utf-8")
)
_LIVE_HIT = {
    "id": "1857641",
    "metadata": {
        "control_number": 1857641,
        "titles": [{"title": "Binary black hole merger gravitational waves"}],
        "abstracts": {"value": "We study binary black hole mergers detected by LIGO."},
        "authors": [{"full_name": "Jane Physicist"}],
        "publication_info": [{"journal_title": "Physical Review D", "year": 2016}],
        "arxiv_eprints": [{"value": "1602.03837", "categories": ["gr-qc"]}],
        "citation_count": 42,
    },
}


class TestInspireHepAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_entry_maps_metadata(self) -> None:
        entry = _SEARCH["results"][0]
        row = inspire_hep._row_from_entry(entry)
        self.assertEqual(row["_adapter"], "inspire_hep")
        self.assertIn("GW150914", row["title"])
        self.assertEqual(row["inspire_recid"], "1403512")

    def test_row_from_inspire_hit_prefers_arxiv_url(self) -> None:
        row = inspire_hep._row_from_inspire_hit(_LIVE_HIT)
        assert row is not None
        self.assertEqual(row["_adapter"], "inspire_hep")
        self.assertIn("arxiv.org/abs/1602.03837", row["url"] or "")
        self.assertEqual(row["citation_count"], 42)
        self.assertTrue(row["full_text"])

    @patch.object(inspire_hep, "fetch_search_results", return_value=_SEARCH)
    def test_search_fixture(self, _mock_fetch) -> None:
        rows = inspire_hep.search_inspire_hep(
            "gravitational wave detection LIGO binary black hole",
            max_results=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "inspire_hep")

    @patch.object(
        inspire_hep,
        "_fetch_inspire_live",
        return_value={"hits": {"hits": [_LIVE_HIT]}},
    )
    def test_live_search(self, _mock_live) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)
        rows = inspire_hep.search_inspire_hep("LIGO binary black hole", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["inspire_recid"], "1857641")


if __name__ == "__main__":
    unittest.main()
