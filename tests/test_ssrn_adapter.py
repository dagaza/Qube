"""Tests for SSRN adapter (OpenAlex-backed)."""

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

from core.knowledge.adapters import ssrn  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SSRN = json.loads((_FIXTURES / "ssrn_search_taylor_rule.json").read_text(encoding="utf-8"))


class TestSsrnAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(ssrn, "fetch_search_results", return_value=_SSRN)
    def test_fixture_search_returns_rows(self, _mock_fetch) -> None:
        rows = ssrn.search_ssrn(
            "central bank inflation targeting Taylor rule empirical",
            max_results=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "ssrn")
        self.assertIn("Taylor", rows[0]["title"])

    @patch.object(ssrn, "fetch_search_results", return_value={"results": []})
    def test_live_search_empty_when_no_hits(self, _mock_fetch) -> None:
        rows = ssrn.search_ssrn("Taylor rule inflation targeting")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
