"""Tests for SSRN adapter stub (Phase 6c-4)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters import ssrn  # noqa: E402


class TestSsrnAdapter(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_live_search_returns_empty_without_fixtures(self) -> None:
        rows = ssrn.search_ssrn("Taylor rule inflation targeting")
        self.assertEqual(rows, [])

    def test_fixture_search_returns_rows(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"
        rows = ssrn.search_ssrn(
            "central bank inflation targeting Taylor rule empirical",
            max_results=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "ssrn")
        self.assertIn("Taylor", rows[0]["title"])


if __name__ == "__main__":
    unittest.main()
