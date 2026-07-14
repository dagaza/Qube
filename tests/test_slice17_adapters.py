"""Tests for Slice 17 adapters (Congress.gov, GovInfo, legislation.gov.uk)."""

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

from core.knowledge.adapters import congress_gov, govinfo, legislation_uk  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.bundle_builder import _legal_row_to_evidence  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_SLICE17_ADAPTER_IDS = ("congress_gov", "govinfo", "legislation_uk")


class TestSlice17Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _SLICE17_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_keyed_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        for provider_id in ("congress_gov", "govinfo"):
            self.assertIn(provider_id, active)

    def test_legislation_uk_has_no_required_hint(self) -> None:
        self.assertIsNone(adapter_credentials_hint("legislation_uk"))

    def test_congress_gov_requires_credentials(self) -> None:
        hint = adapter_credentials_hint("congress_gov")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())
        spec = get_provider_credential_spec("congress_gov")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.key_required)

    def test_govinfo_requires_credentials(self) -> None:
        hint = adapter_credentials_hint("govinfo")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())


class TestSlice17AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_congress_gov_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "congress_gov_search_privacy.json").read_text(encoding="utf-8")
        )
        with patch.object(congress_gov, "fetch_search_results", return_value=fixture):
            rows = congress_gov.search_congress_gov("privacy rights act")
        self.assertEqual(rows[0]["_adapter"], "congress_gov")
        self.assertEqual(rows[0]["document_type"], "federal_bill")
        ev = _legal_row_to_evidence(rows[0], index=1, retrieved_at=0.0)
        self.assertEqual(ev.adapter, "congress_gov")

    def test_govinfo_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "govinfo_search_privacy_act.json").read_text(encoding="utf-8")
        )
        with patch.object(govinfo, "fetch_search_results", return_value=fixture):
            rows = govinfo.search_govinfo("privacy act")
        self.assertEqual(rows[0]["_adapter"], "govinfo")
        self.assertEqual(rows[0]["document_type"], "federal_statute")

    def test_legislation_uk_fixture_search(self) -> None:
        fixture = {
            "feed": (_FIXTURES / "legislation_uk_search_data_protection.xml").read_text(
                encoding="utf-8"
            )
        }
        with patch.object(legislation_uk, "fetch_search_results", return_value=fixture):
            rows = legislation_uk.search_legislation_uk("Data Protection Act")
        self.assertEqual(rows[0]["_adapter"], "legislation_uk")
        self.assertEqual(rows[0]["document_type"], "uk_legislation")
        self.assertIn("legislation.gov.uk", rows[0]["url"])


if __name__ == "__main__":
    unittest.main()
