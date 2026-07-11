"""Tests for Slice 8 adapters (EUR-Lex, CanLII, BAILII)."""

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

from core.knowledge.adapters import bailii, canlii, eur_lex  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.bundle_builder import _legal_row_to_evidence  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
    provider_has_implemented_adapter,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_EUR = json.loads((_FIXTURES / "eur_lex_search_gdpr.json").read_text(encoding="utf-8"))
_CANLII = json.loads((_FIXTURES / "canlii_search_charter.json").read_text(encoding="utf-8"))
_BAILII_HTML = (_FIXTURES / "bailii_search_privacy.html").read_text(encoding="utf-8")


class TestSlice8Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in ("eur_lex", "canlii", "bailii"):
            self.assertIsNotNone(get_search_function(adapter_id))

    def test_canlii_provider_active(self) -> None:
        active = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("canlii", active)
        spec = get_provider_credential_spec("canlii")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(provider_has_implemented_adapter(spec))

    def test_eur_lex_has_no_credentials_hint(self) -> None:
        self.assertIsNone(adapter_credentials_hint("eur_lex"))

    def test_bailii_has_no_credentials_hint(self) -> None:
        self.assertIsNone(adapter_credentials_hint("bailii"))

    def test_canlii_requires_key_hint(self) -> None:
        hint = adapter_credentials_hint("canlii")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())


class TestEurLexAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(eur_lex, "fetch_search_results", return_value=_EUR)
    def test_search_eur_lex_fixture(self, _mock_fetch) -> None:
        rows = eur_lex.search_eur_lex("GDPR personal data", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "eur_lex")
        self.assertEqual(rows[0]["celex"], "32016R0679")
        self.assertIn("eur-lex.europa.eu", rows[0]["url"])

    @patch.object(eur_lex, "fetch_search_results", return_value=_EUR)
    def test_legal_evidence_mapping(self, _mock_fetch) -> None:
        rows = eur_lex.search_eur_lex("GDPR", max_results=1)
        ev = _legal_row_to_evidence(rows[0], index=1, retrieved_at=0.0)
        self.assertEqual(ev.adapter, "eur_lex")
        self.assertEqual(ev.document_type, "eu_legal_act")
        self.assertEqual(ev.venue, "EUR-Lex")


class TestCanliiAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(canlii, "fetch_search_results", return_value=_CANLII)
    def test_search_canlii_fixture(self, _mock_fetch) -> None:
        rows = canlii.search_canlii("Charter constitutional amendment", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "canlii")
        self.assertIn("Constitution", rows[0]["title"])
        self.assertTrue(rows[0]["url"])

    @patch.object(canlii, "fetch_search_results", return_value=_CANLII)
    def test_legal_evidence_mapping(self, _mock_fetch) -> None:
        rows = canlii.search_canlii("Charter", max_results=1)
        ev = _legal_row_to_evidence(rows[0], index=1, retrieved_at=0.0)
        self.assertEqual(ev.adapter, "canlii")
        self.assertEqual(ev.document_type, "court_opinion")
        self.assertEqual(ev.venue, "CanLII")


class TestBailiiAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    @patch.object(bailii, "fetch_search_html", return_value=_BAILII_HTML)
    def test_search_bailii_fixture(self, _mock_fetch) -> None:
        rows = bailii.search_bailii("Donoghue v Stevenson", max_results=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["_adapter"], "bailii")
        self.assertIn("bailii.org", rows[0]["url"])

    def test_rows_from_html_parser(self) -> None:
        rows = bailii._rows_from_html(_BAILII_HTML, max_results=3)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(rows[0]["title"])


if __name__ == "__main__":
    unittest.main()
