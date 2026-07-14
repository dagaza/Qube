"""Tests for Tier 1–2 adapters (SSRN, NOAA, PsyArXiv, NASA Earthdata)."""

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
    nasa_earthdata,
    noaa,
    psyarxiv,
    ssrn,
)
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT,
    detect_scientific_discipline,
    preferred_adapters_for_discipline,
)
from core.knowledge.scientific_discipline_packs import get_discipline_pack  # noqa: E402

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SSRN_FIX = json.loads((_FIXTURES / "ssrn_search_taylor_rule.json").read_text(encoding="utf-8"))
_NOAA = json.loads((_FIXTURES / "noaa_search_temperature.json").read_text(encoding="utf-8"))
_PSY = json.loads((_FIXTURES / "psyarxiv_search_cognitive_load.json").read_text(encoding="utf-8"))
_NASA = json.loads((_FIXTURES / "nasa_earthdata_search_sst.json").read_text(encoding="utf-8"))


class TestTier1Tier2Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in ("ssrn", "noaa", "psyarxiv", "nasa_earthdata"):
            self.assertIsNotNone(get_search_function(adapter_id))

    def test_noaa_provider_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("noaa", active)
        spec = get_provider_credential_spec("noaa")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.key_required)

    def test_ssrn_and_psyarxiv_no_credentials(self) -> None:
        self.assertIsNone(adapter_credentials_hint("ssrn"))
        self.assertIsNone(adapter_credentials_hint("psyarxiv"))
        self.assertIsNone(adapter_credentials_hint("nasa_earthdata"))

    def test_earth_environment_pack_active(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT)
        self.assertIsNotNone(pack)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT)
        self.assertIn("noaa", order)
        self.assertIn("nasa_earthdata", order)

    def test_earth_discipline_detection(self) -> None:
        match = detect_scientific_discipline(
            "satellite sea surface temperature climate reanalysis dataset"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT)


class TestSsrnAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(ssrn, "fetch_search_results", return_value=_SSRN_FIX)
    def test_search_ssrn_fixture(self, _mock_fetch) -> None:
        rows = ssrn.search_ssrn("Taylor rule inflation targeting", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "ssrn")
        self.assertIn("Taylor", rows[0]["title"])


class TestNoaaAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(noaa, "fetch_search_results", return_value=_NOAA)
    def test_search_noaa_fixture(self, _mock_fetch) -> None:
        rows = noaa.search_noaa("global temperature daily", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "noaa")
        self.assertIn("GHCND", rows[0]["title"] + str(rows[0].get("dataset_id")))


class TestPsyarxivAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(psyarxiv, "fetch_search_results", return_value=_PSY)
    def test_search_psyarxiv_fixture(self, _mock_fetch) -> None:
        rows = psyarxiv.search_psyarxiv("working memory cognitive load", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "psyarxiv")
        self.assertIn("memory", rows[0]["title"].lower())


class TestNasaEarthdataAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(nasa_earthdata, "fetch_search_results", return_value=_NASA)
    def test_search_nasa_earthdata_fixture(self, _mock_fetch) -> None:
        rows = nasa_earthdata.search_nasa_earthdata("sea surface temperature", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "nasa_earthdata")
        self.assertIn("Sea Surface", rows[0]["title"])


if __name__ == "__main__":
    unittest.main()
