"""Tests for Slice 7a bibliographic adapters (Crossref, Semantic Scholar, NASA ADS)."""

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

from core.knowledge.adapters import crossref, nasa_ads, semantic_scholar  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    list_active_provider_credential_specs,
    provider_has_implemented_adapter,
    get_provider_credential_spec,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_CROSSREF = json.loads((_FIXTURES / "crossref_search_climate.json").read_text(encoding="utf-8"))
_SEMANTIC = json.loads(
    (_FIXTURES / "semantic_scholar_search_transformer.json").read_text(encoding="utf-8")
)
_NASA = json.loads((_FIXTURES / "nasa_ads_search_ligo.json").read_text(encoding="utf-8"))


class TestSlice7aRegistry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in ("crossref", "semantic_scholar", "nasa_ads"):
            self.assertIsNotNone(get_search_function(adapter_id))

    def test_keyed_providers_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("semantic_scholar", active)
        self.assertIn("nasa_ads", active)

    def test_semantic_scholar_requires_key_hint(self) -> None:
        hint = adapter_credentials_hint("semantic_scholar")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())

    def test_crossref_has_no_credentials_hint(self) -> None:
        self.assertIsNone(adapter_credentials_hint("crossref"))

    def test_implemented_flags(self) -> None:
        for provider_id in ("semantic_scholar", "nasa_ads"):
            spec = get_provider_credential_spec(provider_id)
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertTrue(provider_has_implemented_adapter(spec))


class TestCrossrefAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_row_from_item_maps_doi(self) -> None:
        item = _CROSSREF["message"]["items"][0]
        row = crossref._row_from_item(item)
        assert row is not None
        self.assertEqual(row["_adapter"], "crossref")
        self.assertIn("Arctic sea ice", row["title"])

    @patch.object(crossref, "fetch_search_results", return_value=_CROSSREF)
    def test_search_crossref_fixture(self, _mock_fetch) -> None:
        rows = crossref.search_crossref("Arctic sea ice climate", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "crossref")


class TestSemanticScholarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(semantic_scholar, "fetch_search_results", return_value=_SEMANTIC)
    def test_search_semantic_scholar_fixture(self, _mock_fetch) -> None:
        rows = semantic_scholar.search_semantic_scholar(
            "transformer attention mechanism",
            max_results=2,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "semantic_scholar")
        self.assertIn("Attention", rows[0]["title"])


class TestNasaAdsAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(nasa_ads, "fetch_search_results", return_value=_NASA)
    def test_search_nasa_ads_fixture(self, _mock_fetch) -> None:
        rows = nasa_ads.search_nasa_ads("LIGO gravitational waves", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "nasa_ads")
        self.assertIn("Gravitational Waves", rows[0]["title"])


if __name__ == "__main__":
    unittest.main()
