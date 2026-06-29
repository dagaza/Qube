"""Tests for RePEc adapter (Phase 6 Slice 6b)."""

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

from core.knowledge.adapters import repec  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SEARCH = json.loads(
    (_FIXTURES / "repec_search_monetary.json").read_text(encoding="utf-8")
)


class TestRepecAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_entry_maps_metadata(self) -> None:
        entry = _SEARCH["results"][0]
        row = repec._row_from_entry(entry)
        self.assertEqual(row["_adapter"], "repec")
        self.assertIn("Monetary Policy", row["title"])
        self.assertEqual(row["repec_handle"], "RePEc:example:mpivar2020")
        self.assertTrue(row["full_text"])

    @patch.object(repec, "fetch_search_results", return_value=_SEARCH)
    def test_search_repec_fixture(self, _mock_fetch) -> None:
        rows = repec.search_repec(
            "monetary policy inflation econometric VAR model",
            max_results=2,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "repec")

    def test_live_search_returns_empty_without_api_key(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)
        os.environ.pop("QUBE_REPEC_API_KEY", None)
        rows = repec.search_repec("monetary policy inflation")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
