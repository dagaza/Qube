"""Tests for Slice 7c adapters (Companies House, Alpha Vantage)."""

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

from core.knowledge.adapters import alpha_vantage, companies_house  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.bundle_builder import _finance_row_to_evidence  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
    provider_has_implemented_adapter,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_CH = json.loads((_FIXTURES / "companies_house_search_tesco.json").read_text(encoding="utf-8"))
_AV = json.loads((_FIXTURES / "alpha_vantage_search_microsoft.json").read_text(encoding="utf-8"))


class TestSlice7cRegistry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in ("companies_house", "alpha_vantage"):
            self.assertIsNotNone(get_search_function(adapter_id))

    def test_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("companies_house", active)
        self.assertIn("alpha_vantage", active)

    def test_companies_house_requires_key_hint(self) -> None:
        spec = get_provider_credential_spec("companies_house")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(provider_has_implemented_adapter(spec))
        hint = adapter_credentials_hint("companies_house")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())

    def test_alpha_vantage_requires_key_hint(self) -> None:
        hint = adapter_credentials_hint("alpha_vantage")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())


class TestCompaniesHouseAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(companies_house, "fetch_search_results", return_value=_CH)
    def test_search_companies_house_fixture(self, _mock_fetch) -> None:
        rows = companies_house.search_companies_house("Tesco PLC", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "companies_house")
        self.assertEqual(rows[0]["company_number"], "00445790")
        self.assertIn("TESCO", rows[0]["title"].upper())

    @patch.object(companies_house, "fetch_search_results", return_value=_CH)
    def test_finance_evidence_mapping(self, _mock_fetch) -> None:
        rows = companies_house.search_companies_house("Tesco", max_results=1)
        ev = _finance_row_to_evidence(rows[0], index=1, retrieved_at=0.0)
        self.assertEqual(ev.adapter, "companies_house")
        self.assertEqual(ev.document_type, "uk_company_registry")
        self.assertEqual(ev.venue, "Companies House")
        self.assertEqual((ev.raw_metadata or {}).get("company_number"), "00445790")


class TestAlphaVantageAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(alpha_vantage, "fetch_search_results", return_value=_AV)
    def test_search_alpha_vantage_fixture(self, _mock_fetch) -> None:
        rows = alpha_vantage.search_alpha_vantage("Microsoft", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "alpha_vantage")
        self.assertEqual(rows[0]["symbol"], "MSFT")
        self.assertIn("Microsoft", rows[0]["title"])

    @patch.object(alpha_vantage, "fetch_search_results", return_value=_AV)
    def test_finance_evidence_mapping(self, _mock_fetch) -> None:
        rows = alpha_vantage.search_alpha_vantage("Microsoft", max_results=1)
        ev = _finance_row_to_evidence(rows[0], index=1, retrieved_at=0.0)
        self.assertEqual(ev.adapter, "alpha_vantage")
        self.assertEqual(ev.document_type, "market_symbol")
        self.assertEqual(ev.venue, "Alpha Vantage")
        self.assertEqual((ev.raw_metadata or {}).get("symbol"), "MSFT")


if __name__ == "__main__":
    unittest.main()
