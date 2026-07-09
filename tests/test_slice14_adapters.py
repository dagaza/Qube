"""Tests for Slice 14 adapters (IPCC, FAO, USDA, Copernicus CDS)."""

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

from core.knowledge.adapters import (  # noqa: E402
    copernicus_cds,
    fao,
    ipcc,
    usda,
)
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT,
    preferred_adapters_for_discipline,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_SLICE14_ADAPTER_IDS = ("ipcc", "fao", "usda", "copernicus_cds")


class TestSlice14Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _SLICE14_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_keyed_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        for provider_id in ("fao", "usda", "copernicus_cds"):
            self.assertIn(provider_id, active)

    def test_anonymous_adapters_have_no_required_hint(self) -> None:
        self.assertIsNone(adapter_credentials_hint("ipcc"))

    def test_fao_requires_credentials(self) -> None:
        hint = adapter_credentials_hint("fao")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("Provider credentials", hint)
        spec = get_provider_credential_spec("fao")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.key_required)

    def test_discipline_pack_updates(self) -> None:
        earth_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT)
        self.assertIn("ipcc", earth_order)
        self.assertIn("copernicus_cds", earth_order)


class TestSlice14AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_ipcc_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "ipcc_search_sea_level.json").read_text(encoding="utf-8")
        )
        with patch.object(ipcc, "fetch_search_results", return_value=fixture):
            rows = ipcc.search_ipcc("sea level")
        self.assertEqual(rows[0]["_adapter"], "ipcc")
        self.assertEqual(rows[0]["document_type"], "assessment_report")

    def test_fao_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "fao_search_wheat.json").read_text(encoding="utf-8")
        )
        with patch.object(fao, "fetch_search_results", return_value=fixture):
            rows = fao.search_fao("wheat")
        self.assertEqual(rows[0]["_adapter"], "fao")
        self.assertEqual(rows[0]["document_type"], "agricultural_dataset")

    def test_usda_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "usda_search_wheat.json").read_text(encoding="utf-8")
        )
        with patch.object(usda, "fetch_search_results", return_value=fixture):
            rows = usda.search_usda("wheat")
        self.assertEqual(rows[0]["_adapter"], "usda")
        self.assertEqual(rows[0]["document_type"], "agricultural_indicator")

    def test_copernicus_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "copernicus_cds_search_temperature.json").read_text(encoding="utf-8")
        )
        with patch.object(copernicus_cds, "fetch_search_results", return_value=fixture):
            rows = copernicus_cds.search_copernicus_cds("temperature")
        self.assertEqual(rows[0]["_adapter"], "copernicus_cds")
        self.assertEqual(rows[0]["document_type"], "climate_dataset")


if __name__ == "__main__":
    unittest.main()
