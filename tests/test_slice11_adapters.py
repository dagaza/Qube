"""Tests for Tier 3–4 adapters (ACM DL, PsycINFO, Bloomberg)."""

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

from core.knowledge.adapters import acm_dl, bloomberg_api, psycinfo  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
)
from core.knowledge.scientific_discipline import preferred_adapters_for_discipline  # noqa: E402
from core.knowledge.scientific_discipline_packs import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
    SCIENTIFIC_DISCIPLINE_PSYCHOLOGY,
    get_discipline_pack,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_ACM = json.loads((_FIXTURES / "acm_dl_search_transformer.json").read_text(encoding="utf-8"))
_PSY = json.loads((_FIXTURES / "psycinfo_search_cognitive_load.json").read_text(encoding="utf-8"))
_BBG = json.loads((_FIXTURES / "bloomberg_search_microsoft.json").read_text(encoding="utf-8"))


class TestTier3Tier4Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in ("acm_dl", "psycinfo", "bloomberg_api"):
            self.assertIsNotNone(get_search_function(adapter_id))

    def test_keyed_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("ebsco_eds", active)
        self.assertIn("bloomberg", active)
        ebsco = get_provider_credential_spec("ebsco_eds")
        bloomberg = get_provider_credential_spec("bloomberg")
        self.assertIsNotNone(ebsco)
        self.assertIsNotNone(bloomberg)
        assert ebsco is not None
        assert bloomberg is not None
        self.assertTrue(ebsco.key_required)
        self.assertTrue(bloomberg.key_required)

    def test_acm_dl_no_provider_row(self) -> None:
        self.assertIsNone(adapter_credentials_hint("acm_dl"))

    def test_psycinfo_and_bloomberg_hints(self) -> None:
        psyc_hint = adapter_credentials_hint("psycinfo")
        bbg_hint = adapter_credentials_hint("bloomberg_api")
        self.assertIsNotNone(psyc_hint)
        self.assertIsNotNone(bbg_hint)
        assert psyc_hint is not None
        assert bbg_hint is not None
        self.assertIn("Provider credentials", psyc_hint)
        self.assertIn("Provider credentials", bbg_hint)

    def test_discipline_pack_updates(self) -> None:
        cs = get_discipline_pack(SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        psych = get_discipline_pack(SCIENTIFIC_DISCIPLINE_PSYCHOLOGY)
        self.assertIsNotNone(cs)
        self.assertIsNotNone(psych)
        assert cs is not None
        assert psych is not None
        cs_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        psych_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_PSYCHOLOGY)
        self.assertIn("acm_dl", cs_order)
        self.assertIn("psycinfo", psych_order)


class TestAcmDlAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(acm_dl, "fetch_search_results", return_value=_ACM)
    def test_search_acm_dl_fixture(self, _mock_fetch) -> None:
        rows = acm_dl.search_acm_dl("transformer attention", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "acm_dl")
        self.assertIn("Attention", rows[0]["title"])


class TestPsycinfoAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(psycinfo, "fetch_search_results", return_value=_PSY)
    def test_search_psycinfo_fixture(self, _mock_fetch) -> None:
        rows = psycinfo.search_psycinfo("working memory cognitive load", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "psycinfo")
        self.assertIn("Cognitive Load", rows[0]["title"])


class TestBloombergAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(bloomberg_api, "fetch_search_results", return_value=_BBG)
    def test_search_bloomberg_fixture(self, _mock_fetch) -> None:
        rows = bloomberg_api.search_bloomberg_api("Microsoft", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "bloomberg_api")
        self.assertIn("MSFT", rows[0]["title"])
