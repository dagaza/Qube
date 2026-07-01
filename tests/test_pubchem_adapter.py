"""Tests for PubChem adapter (Phase 6c-2)."""

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

from core.knowledge.adapters import pubchem  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_ASPIRIN = json.loads(
    (_FIXTURES / "pubchem_search_aspirin.json").read_text(encoding="utf-8")
)


class TestPubchemAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_entry_maps_cid_metadata(self) -> None:
        entry = _ASPIRIN["results"][0]
        row = pubchem._row_from_entry(entry)
        self.assertEqual(row["_adapter"], "pubchem")
        self.assertEqual(row["pubchem_cid"], 2244)
        self.assertEqual(row["molecular_formula"], "C9H8O4")
        self.assertIn("2244", row["url"])

    def test_compound_name_candidates_prefers_phrases(self) -> None:
        names = pubchem._compound_name_candidates(
            "aspirin acetylsalicylic acid binding COX-2 cyclooxygenase"
        )
        self.assertIn("acetylsalicylic acid", names)
        self.assertIn("aspirin", names)

    @patch.object(pubchem, "fetch_search_results", return_value=_ASPIRIN)
    def test_search_pubchem_fixture(self, _mock_fetch) -> None:
        rows = pubchem.search_pubchem(
            "aspirin acetylsalicylic acid binding COX-2",
            max_results=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "pubchem")
        self.assertEqual(rows[0]["pubchem_cid"], 2244)

    @patch.object(pubchem, "_fetch_cid", return_value=2244)
    @patch.object(
        pubchem,
        "_fetch_compound_record",
        return_value=_ASPIRIN["results"][0],
    )
    def test_live_path_builds_row(self, _mock_record, _mock_cid) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)
        rows = pubchem.search_pubchem("aspirin", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pubchem_cid"], 2244)


if __name__ == "__main__":
    unittest.main()
