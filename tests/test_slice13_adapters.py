"""Tests for Slice 13 adapters (OECD, NICE, CDC, WHO)."""

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

from core.knowledge.adapters import cdc, nice, oecd, who  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_MEDICINE,
    preferred_adapters_for_discipline,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_SLICE13_ADAPTER_IDS = ("oecd", "nice", "cdc", "who")


class TestSlice13Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _SLICE13_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_nice_provider_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("nice", active)

    def test_anonymous_adapters_have_no_required_hint(self) -> None:
        for adapter_id in ("oecd", "cdc", "who"):
            self.assertIsNone(adapter_credentials_hint(adapter_id))

    def test_nice_requires_credentials(self) -> None:
        hint = adapter_credentials_hint("nice")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("Configure", hint)
        spec = get_provider_credential_spec("nice")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.key_required)

    def test_discipline_pack_updates(self) -> None:
        med_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_MEDICINE)
        econ_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_ECONOMICS)
        self.assertIn("nice", med_order)
        self.assertIn("cdc", med_order)
        self.assertIn("who", med_order)
        self.assertIn("oecd", econ_order)


class TestSlice13AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_oecd_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "oecd_search_unemployment.json").read_text(encoding="utf-8")
        )
        with patch.object(oecd, "fetch_search_results", return_value=fixture):
            rows = oecd.search_oecd("unemployment")
        self.assertEqual(rows[0]["_adapter"], "oecd")
        self.assertEqual(rows[0]["document_type"], "statistical_release")

    def test_nice_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "nice_search_hypertension.json").read_text(encoding="utf-8")
        )
        with patch.object(nice, "fetch_search_results", return_value=fixture):
            rows = nice.search_nice("hypertension")
        self.assertEqual(rows[0]["_adapter"], "nice")
        self.assertEqual(rows[0]["document_type"], "clinical_guideline")
        self.assertIn("NG136", rows[0]["title"])

    def test_cdc_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "cdc_search_diabetes.json").read_text(encoding="utf-8")
        )
        with patch.object(cdc, "fetch_search_results", return_value=fixture):
            rows = cdc.search_cdc("diabetes")
        self.assertEqual(rows[0]["_adapter"], "cdc")
        self.assertEqual(rows[0]["document_type"], "health_guidance")

    def test_who_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "who_search_hypertension.json").read_text(encoding="utf-8")
        )
        with patch.object(who, "fetch_search_results", return_value=fixture):
            rows = who.search_who("hypertension")
        self.assertEqual(rows[0]["_adapter"], "who")
        self.assertEqual(rows[0]["document_type"], "health_indicator")


if __name__ == "__main__":
    unittest.main()
