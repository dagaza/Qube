"""Tests for P0 institutional adapters (Slice 12)."""

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
    bls,
    clinicaltrials_gov,
    eurostat,
    ieee_xplore,
    ietf_rfc,
    nist,
    openfda,
    us_census,
    usda_fdc,
    usgs,
    world_bank,
)
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_ENGINEERING,
    SCIENTIFIC_DISCIPLINE_MEDICINE,
    preferred_adapters_for_discipline,
)
from core.knowledge.scientific_discipline_packs import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT,
    get_discipline_pack,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_P0_ADAPTER_IDS = (
    "clinicaltrials_gov",
    "openfda",
    "world_bank",
    "eurostat",
    "usgs",
    "usda_fdc",
    "nist",
    "ietf_rfc",
    "bls",
    "us_census",
    "ieee_xplore",
)


class TestP0Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _P0_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_keyed_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        for provider_id in ("usda_fdc", "bls", "us_census", "nist", "ieee_xplore"):
            self.assertIn(provider_id, active)

    def test_anonymous_adapters_have_no_required_hint(self) -> None:
        for adapter_id in (
            "clinicaltrials_gov",
            "openfda",
            "world_bank",
            "eurostat",
            "usgs",
            "ietf_rfc",
        ):
            self.assertIsNone(adapter_credentials_hint(adapter_id))

    def test_keyed_adapter_hints(self) -> None:
        bls_hint = adapter_credentials_hint("bls")
        ieee_hint = adapter_credentials_hint("ieee_xplore")
        self.assertIsNotNone(bls_hint)
        self.assertIsNotNone(ieee_hint)
        assert bls_hint is not None
        assert ieee_hint is not None
        self.assertIn("Configure", bls_hint)
        self.assertIn("Configure", ieee_hint)

    def test_discipline_pack_updates(self) -> None:
        medicine = get_discipline_pack(SCIENTIFIC_DISCIPLINE_MEDICINE)
        earth = get_discipline_pack(SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT)
        engineering = get_discipline_pack(SCIENTIFIC_DISCIPLINE_ENGINEERING)
        self.assertIsNotNone(medicine)
        self.assertIsNotNone(earth)
        self.assertIsNotNone(engineering)
        assert medicine is not None
        assert earth is not None
        assert engineering is not None
        med_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_MEDICINE)
        econ_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_ECONOMICS)
        eng_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_ENGINEERING)
        self.assertIn("clinicaltrials_gov", med_order)
        self.assertIn("world_bank", econ_order)
        self.assertIn("ieee_xplore", eng_order)

    def test_provider_specs(self) -> None:
        ieee = get_provider_credential_spec("ieee_xplore")
        self.assertIsNotNone(ieee)
        assert ieee is not None
        self.assertTrue(ieee.key_required)


class TestP0AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_clinicaltrials_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "clinicaltrials_gov_search_diabetes.json").read_text(encoding="utf-8")
        )
        with patch.object(clinicaltrials_gov, "fetch_search_results", return_value=fixture):
            rows = clinicaltrials_gov.search_clinicaltrials_gov("diabetes")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "clinicaltrials_gov")
        self.assertIn("NCT", rows[0]["title"])

    def test_openfda_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "openfda_search_hypertension.json").read_text(encoding="utf-8")
        )
        with patch.object(openfda, "fetch_search_results", return_value=fixture):
            rows = openfda.search_openfda("hypertension")
        self.assertEqual(rows[0]["_adapter"], "openfda")
        self.assertEqual(rows[0]["document_type"], "regulatory_label")

    def test_world_bank_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "world_bank_search_unemployment.json").read_text(encoding="utf-8")
        )
        with patch.object(world_bank, "fetch_search_results", return_value=fixture):
            rows = world_bank.search_world_bank("unemployment")
        self.assertEqual(rows[0]["document_type"], "statistical_indicator")

    def test_eurostat_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "eurostat_search_unemployment.json").read_text(encoding="utf-8")
        )
        with patch.object(eurostat, "fetch_search_results", return_value=fixture):
            rows = eurostat.search_eurostat("unemployment")
        self.assertEqual(rows[0]["_adapter"], "eurostat")

    def test_usgs_fixture_search(self) -> None:
        fixture = json.loads((_FIXTURES / "usgs_search_earthquake.json").read_text(encoding="utf-8"))
        with patch.object(usgs, "fetch_search_results", return_value=fixture):
            rows = usgs.search_usgs("earthquake")
        self.assertEqual(rows[0]["document_type"], "government_publication")

    def test_usda_fdc_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "usda_fdc_search_apple.json").read_text(encoding="utf-8")
        )
        with patch.object(usda_fdc, "fetch_search_results", return_value=fixture):
            rows = usda_fdc.search_usda_fdc("apple")
        self.assertEqual(rows[0]["document_type"], "nutrition_dataset")

    def test_nist_fixture_search(self) -> None:
        fixture = json.loads((_FIXTURES / "nist_search_encryption.json").read_text(encoding="utf-8"))
        with patch.object(nist, "fetch_search_results", return_value=fixture):
            rows = nist.search_nist("encryption")
        self.assertEqual(rows[0]["document_type"], "standard_reference")

    def test_ietf_fixture_search(self) -> None:
        fixture = json.loads((_FIXTURES / "ietf_rfc_search_tls.json").read_text(encoding="utf-8"))
        with patch.object(ietf_rfc, "fetch_search_results", return_value=fixture):
            rows = ietf_rfc.search_ietf_rfc("tls")
        self.assertEqual(rows[0]["document_type"], "standard_document")

    def test_bls_fixture_search(self) -> None:
        fixture = json.loads((_FIXTURES / "bls_search_unemployment.json").read_text(encoding="utf-8"))
        with patch.object(bls, "fetch_search_results", return_value=fixture):
            rows = bls.search_bls("unemployment")
        self.assertEqual(rows[0]["_adapter"], "bls")

    def test_us_census_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "us_census_search_population.json").read_text(encoding="utf-8")
        )
        with patch.object(us_census, "fetch_search_results", return_value=fixture):
            rows = us_census.search_us_census("population")
        self.assertEqual(rows[0]["document_type"], "statistical_release")

    def test_ieee_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "ieee_xplore_search_robotics.json").read_text(encoding="utf-8")
        )
        with patch.object(ieee_xplore, "fetch_search_results", return_value=fixture):
            rows = ieee_xplore.search_ieee_xplore("robotics")
        self.assertEqual(rows[0]["_adapter"], "ieee_xplore")


if __name__ == "__main__":
    unittest.main()
