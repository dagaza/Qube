"""Tests for Slice 7b adapters (SocArXiv, FRED, Europe PMC)."""

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

from core.knowledge.adapters import europe_pmc, fred, socarxiv  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
    provider_has_implemented_adapter,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_EUROPE = json.loads((_FIXTURES / "europe_pmc_search_trials.json").read_text(encoding="utf-8"))
_SOCARXIV = json.loads((_FIXTURES / "socarxiv_search_inequality.json").read_text(encoding="utf-8"))
_FRED = json.loads((_FIXTURES / "fred_search_unemployment.json").read_text(encoding="utf-8"))


class TestSlice7bRegistry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in ("socarxiv", "fred", "europe_pmc"):
            self.assertIsNotNone(get_search_function(adapter_id))

    def test_fred_provider_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("fred", active)
        spec = get_provider_credential_spec("fred")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(provider_has_implemented_adapter(spec))

    def test_socarxiv_has_no_credentials_hint(self) -> None:
        self.assertIsNone(adapter_credentials_hint("socarxiv"))

    def test_fred_requires_key_hint(self) -> None:
        hint = adapter_credentials_hint("fred")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("required", hint.lower())


class TestEuropePmcAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(europe_pmc, "fetch_search_results", return_value=_EUROPE)
    def test_search_europe_pmc_fixture(self, _mock_fetch) -> None:
        rows = europe_pmc.search_europe_pmc("semaglutide cardiovascular trial", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "europe_pmc")


class TestSocArxivAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(socarxiv, "fetch_search_results", return_value=_SOCARXIV)
    def test_search_socarxiv_fixture(self, _mock_fetch) -> None:
        rows = socarxiv.search_socarxiv("income inequality sociology", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "socarxiv")
        self.assertIn("inequality", rows[0]["title"].lower())


class TestFredAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(fred, "fetch_search_results", return_value=_FRED)
    def test_search_fred_fixture(self, _mock_fetch) -> None:
        rows = fred.search_fred("unemployment rate", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "fred")
        self.assertEqual(rows[0]["series_id"], "UNRATE")


if __name__ == "__main__":
    unittest.main()
