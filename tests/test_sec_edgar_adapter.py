"""Tests for SEC EDGAR adapter (Phase 6 Slice 5a)."""

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

from core.knowledge.adapters import sec_edgar  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_TICKERS = json.loads((_FIXTURES / "sec_company_tickers_mini.json").read_text(encoding="utf-8"))
_SUBMISSIONS = json.loads((_FIXTURES / "sec_submissions_aapl.json").read_text(encoding="utf-8"))


class TestSecEdgarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        sec_edgar._tickers_cache = None
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)
        sec_edgar._tickers_cache = None

    def test_resolve_company_by_ticker(self) -> None:
        company = sec_edgar.resolve_company("AMZN SEC filings", _TICKERS)
        self.assertIsNotNone(company)
        assert company is not None
        self.assertEqual(company["ticker"], "AMZN")
        self.assertEqual(company["cik"], 1018724)

    def test_resolve_company_by_name(self) -> None:
        company = sec_edgar.resolve_company("Apple Inc annual report", _TICKERS)
        self.assertIsNotNone(company)
        assert company is not None
        self.assertEqual(company["ticker"], "AAPL")

    def test_rows_from_submissions(self) -> None:
        rows = sec_edgar._rows_from_submissions(
            _SUBMISSIONS,
            company_name="Apple Inc.",
            cik=320193,
            form_filter=("10-K",),
            max_results=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["form"], "10-K")
        self.assertIn("sec.gov", rows[0]["url"])

    @patch.object(sec_edgar, "fetch_submissions", return_value=_SUBMISSIONS)
    def test_search_sec_edgar_end_to_end(self, _mock_fetch) -> None:
        rows = sec_edgar.search_sec_edgar("AAPL 10-K", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "sec_edgar")
        self.assertIn("Apple", rows[0]["title"])


if __name__ == "__main__":
    unittest.main()
