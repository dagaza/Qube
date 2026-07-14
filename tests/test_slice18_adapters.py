"""Tests for Slice 18 adapters (USPTO PatentsView, EPO Espacenet)."""

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

from core.knowledge.adapters import epo_espacenet, uspto_patentsview  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_CHEMISTRY,
    SCIENTIFIC_DISCIPLINE_ENGINEERING,
    preferred_adapters_for_discipline,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_SLICE18_ADAPTER_IDS = ("uspto_patentsview", "epo_espacenet")


class TestSlice18Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _SLICE18_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_keyed_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        for provider_id in ("patentsview", "epo_ops"):
            self.assertIn(provider_id, active)

    def test_patentsview_requires_credentials(self) -> None:
        hint = adapter_credentials_hint("uspto_patentsview")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())
        spec = get_provider_credential_spec("patentsview")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.key_required)

    def test_epo_requires_credentials(self) -> None:
        hint = adapter_credentials_hint("epo_espacenet")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())

    def test_discipline_pack_updates(self) -> None:
        engineering_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_ENGINEERING)
        chemistry_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_CHEMISTRY)
        self.assertIn("uspto_patentsview", engineering_order)
        self.assertIn("epo_espacenet", engineering_order)
        self.assertIn("uspto_patentsview", chemistry_order)
        self.assertIn("epo_espacenet", chemistry_order)


class TestSlice18AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_patentsview_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "uspto_patentsview_search_battery.json").read_text(encoding="utf-8")
        )
        with patch.object(uspto_patentsview, "fetch_search_results", return_value=fixture):
            rows = uspto_patentsview.search_uspto_patentsview("lithium battery")
        self.assertEqual(rows[0]["_adapter"], "uspto_patentsview")
        self.assertEqual(rows[0]["document_type"], "patent")

    def test_epo_fixture_search(self) -> None:
        fixture = {
            "xml": (_FIXTURES / "epo_espacenet_search_battery.xml").read_text(encoding="utf-8")
        }
        with patch.object(epo_espacenet, "fetch_search_results", return_value=fixture):
            rows = epo_espacenet.search_epo_espacenet("lithium battery")
        self.assertEqual(rows[0]["_adapter"], "epo_espacenet")
        self.assertEqual(rows[0]["document_type"], "patent")
        self.assertIn("espacenet", rows[0]["url"])


if __name__ == "__main__":
    unittest.main()
