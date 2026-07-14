"""Tests for knowledge credential resolver (HTTP resilience Slice 2)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters import openalex, pubmed_eutils  # noqa: E402
from core.knowledge.adapters.catalog import get_adapter_entry  # noqa: E402
from core.knowledge.credential_resolver import (  # noqa: E402
    CredentialMode,
    api_key_query_params,
    merge_query_params,
    ncbi_rate_per_sec,
    resolve,
)


class TestCredentialResolver(unittest.TestCase):
    def tearDown(self) -> None:
        for name in ("QUBE_OPENALEX_API_KEY", "QUBE_NCBI_API_KEY"):
            os.environ.pop(name, None)

    def test_resolve_anonymous_when_env_unset(self) -> None:
        cred = resolve("openalex")
        self.assertIsNone(cred.api_key)
        self.assertEqual(cred.mode, CredentialMode.ANONYMOUS)

    def test_resolve_openalex_env_key(self) -> None:
        os.environ["QUBE_OPENALEX_API_KEY"] = "test-openalex-key"
        cred = resolve("openalex")
        self.assertEqual(cred.api_key, "test-openalex-key")
        self.assertEqual(cred.mode, CredentialMode.ENV_KEY)

    def test_resolve_ncbi_env_key(self) -> None:
        os.environ["QUBE_NCBI_API_KEY"] = "test-ncbi-key"
        cred = resolve("ncbi")
        self.assertEqual(cred.api_key, "test-ncbi-key")
        self.assertEqual(cred.mode, CredentialMode.ENV_KEY)

    def test_api_key_query_params_empty_without_key(self) -> None:
        self.assertEqual(api_key_query_params("openalex"), {})

    def test_merge_query_params_injects_key(self) -> None:
        os.environ["QUBE_OPENALEX_API_KEY"] = "secret"
        merged = merge_query_params({"search": "crispr"}, "openalex")
        self.assertEqual(merged["search"], "crispr")
        self.assertEqual(merged["api_key"], "secret")

    def test_ncbi_rate_per_sec_anonymous(self) -> None:
        self.assertEqual(ncbi_rate_per_sec(), 2.5)

    def test_ncbi_rate_per_sec_with_key(self) -> None:
        os.environ["QUBE_NCBI_API_KEY"] = "test-ncbi-key"
        self.assertEqual(ncbi_rate_per_sec(), 8.0)


class TestAdapterCredentialWiring(unittest.TestCase):
    def tearDown(self) -> None:
        for name in ("QUBE_OPENALEX_API_KEY", "QUBE_NCBI_API_KEY"):
            os.environ.pop(name, None)

    @patch("core.knowledge.adapters.openalex.knowledge_get")
    def test_openalex_injects_api_key(self, mock_get: MagicMock) -> None:
        os.environ["QUBE_OPENALEX_API_KEY"] = "openalex-test"
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        mock_get.return_value = resp
        openalex.search_openalex("quantum dots", max_results=1)
        _url, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "openalex-test")
        self.assertEqual(kwargs["params"]["search"], "quantum dots")

    @patch("core.knowledge.adapters.openalex.knowledge_get")
    def test_openalex_omits_api_key_when_unset(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        mock_get.return_value = resp
        openalex.search_openalex("quantum dots", max_results=1)
        _url, kwargs = mock_get.call_args
        self.assertNotIn("api_key", kwargs["params"])

    @patch("core.knowledge.adapters.pubmed_eutils.knowledge_get")
    def test_pubmed_injects_ncbi_api_key(self, mock_get: MagicMock) -> None:
        os.environ["QUBE_NCBI_API_KEY"] = "ncbi-test"
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = resp
        pubmed_eutils.search_pubmed("heart failure", max_results=1)
        _url, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "ncbi-test")


class TestCatalogOptionalApiKey(unittest.TestCase):
    def test_openalex_optional_api_key_flag(self) -> None:
        entry = get_adapter_entry("openalex")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.requires_api_key)
        self.assertTrue(entry.optional_api_key)

    def test_pubmed_optional_api_key_flag(self) -> None:
        entry = get_adapter_entry("pubmed")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.optional_api_key)

    def test_pubchem_optional_api_key_flag(self) -> None:
        entry = get_adapter_entry("pubchem")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.optional_api_key)


if __name__ == "__main__":
    unittest.main()
