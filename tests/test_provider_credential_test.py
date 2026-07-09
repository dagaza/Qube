"""Tests for provider credential probes and Settings status copy."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.credentials import connection_mode_display  # noqa: E402
from core.knowledge.provider_credential_test import test_provider_credential  # noqa: E402
from core.knowledge.provider_credentials import (  # noqa: E402
    adapter_credentials_hint,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
    provider_has_implemented_adapter,
)


class TestProviderCredentialStatus(unittest.TestCase):
    def test_semantic_scholar_requires_key_display(self) -> None:
        spec = get_provider_credential_spec("semantic_scholar")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(provider_has_implemented_adapter(spec))
        self.assertEqual(
            connection_mode_display("semantic_scholar"),
            "API key required",
        )

    def test_active_provider_specs_include_slice7a(self) -> None:
        active_ids = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("openalex", active_ids)
        self.assertIn("ncbi", active_ids)
        self.assertIn("courtlistener", active_ids)
        self.assertIn("semantic_scholar", active_ids)
        self.assertIn("nasa_ads", active_ids)
        self.assertIn("fred", active_ids)
        self.assertIn("companies_house", active_ids)
        self.assertIn("alpha_vantage", active_ids)
        self.assertIn("canlii", active_ids)
        self.assertIn("noaa", active_ids)

    def test_fred_requires_key_display(self) -> None:
        self.assertEqual(connection_mode_display("fred"), "API key required")

    def test_adapter_credentials_hint_for_semantic_scholar(self) -> None:
        hint = adapter_credentials_hint("semantic_scholar")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("Configure", hint)

    def test_active_provider_specs_include_slice11(self) -> None:
        active_ids = {spec.provider_id for spec in list_active_provider_credential_specs()}
        self.assertIn("ebsco_eds", active_ids)
        self.assertIn("bloomberg", active_ids)

    def test_adapter_credentials_hint_for_bloomberg(self) -> None:
        hint = adapter_credentials_hint("bloomberg_api")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("Configure", hint)

    def test_adapter_credentials_hint_none_for_acm_dl(self) -> None:
        self.assertIsNone(adapter_credentials_hint("acm_dl"))

    def test_adapter_credentials_hint_for_companies_house(self) -> None:
        hint = adapter_credentials_hint("companies_house")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("api key", hint.lower())

    def test_adapter_credentials_hint_for_live_optional_key(self) -> None:
        hint = adapter_credentials_hint("openalex")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("Configure", hint)

    def test_openalex_anonymous_display(self) -> None:
        self.assertEqual(
            connection_mode_display("openalex"),
            "Anonymous access",
        )


class TestProviderCredentialProbes(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_OPENALEX_API_KEY", None)

    @patch("core.knowledge.provider_credential_test.knowledge_get")
    def test_openalex_probe_success(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        mock_get.return_value = resp
        result = test_provider_credential("openalex")
        self.assertTrue(result.ok)
        self.assertIn("succeeded", result.message.lower())

    @patch("core.knowledge.provider_credential_test.knowledge_get")
    def test_openalex_probe_with_override_key(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        mock_get.return_value = resp
        result = test_provider_credential("openalex", override_secret="test-key")
        self.assertTrue(result.ok)
        _url, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "test-key")

    def test_semantic_scholar_requires_key(self) -> None:
        result = test_provider_credential("semantic_scholar")
        self.assertFalse(result.ok)
        self.assertIn("required", result.message.lower())

    def test_fred_requires_key(self) -> None:
        result = test_provider_credential("fred")
        self.assertFalse(result.ok)
        self.assertIn("required", result.message.lower())

    def test_companies_house_requires_key(self) -> None:
        result = test_provider_credential("companies_house")
        self.assertFalse(result.ok)
        self.assertIn("required", result.message.lower())

    def test_alpha_vantage_requires_key(self) -> None:
        result = test_provider_credential("alpha_vantage")
        self.assertFalse(result.ok)
        self.assertIn("required", result.message.lower())

    def test_canlii_requires_key(self) -> None:
        result = test_provider_credential("canlii")
        self.assertFalse(result.ok)
        self.assertIn("required", result.message.lower())

    def test_noaa_requires_key(self) -> None:
        result = test_provider_credential("noaa")
        self.assertFalse(result.ok)
        self.assertIn("required", result.message.lower())


if __name__ == "__main__":
    unittest.main()
