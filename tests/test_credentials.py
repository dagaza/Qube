"""Tests for knowledge credentials store & resolver (HTTP resilience Slice 9)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters import openalex, pubchem, pubmed_eutils  # noqa: E402
from core.knowledge.credentials import (  # noqa: E402
    CredentialBundle,
    CredentialMode,
    clear_provider_api_key,
    resolve_credential,
    set_provider_api_key,
)
from core.knowledge.credential_resolver import (  # noqa: E402
    authorization_token,
    merge_query_params,
    provider_id_for_adapter,
    resolve,
)
from core.knowledge.provider_credentials import get_provider_credential_spec  # noqa: E402


class _FakeSettingsStore:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value


class TestCredentialStore(unittest.TestCase):
    def setUp(self) -> None:
        self._store = _FakeSettingsStore()
        self._patch_store = patch(
            "core.app_settings._store",
            lambda: self._store,
        )
        self._patch_store.start()

    def tearDown(self) -> None:
        self._patch_store.stop()
        for name in (
            "QUBE_OPENALEX_API_KEY",
            "QUBE_NCBI_API_KEY",
            "QUBE_COURTLISTENER_API_TOKEN",
            "QUBE_KNOWLEDGE_FIXTURES",
        ):
            os.environ.pop(name, None)

    def test_provider_specs_registered(self) -> None:
        openalex_spec = get_provider_credential_spec("openalex")
        ncbi_spec = get_provider_credential_spec("ncbi")
        self.assertIsNotNone(openalex_spec)
        self.assertIsNotNone(ncbi_spec)
        assert ncbi_spec is not None
        self.assertIn("pubmed", ncbi_spec.adapter_ids)
        self.assertIn("pubchem", ncbi_spec.adapter_ids)

    def test_adapter_provider_mapping(self) -> None:
        self.assertEqual(provider_id_for_adapter("pubmed"), "ncbi")
        self.assertEqual(provider_id_for_adapter("pubchem"), "ncbi")
        self.assertEqual(provider_id_for_adapter("openalex"), "openalex")

    def test_user_settings_used_when_env_unset(self) -> None:
        set_provider_api_key("openalex", "user-openalex-key")
        cred = resolve_credential("openalex")
        self.assertEqual(cred.secret, "user-openalex-key")
        self.assertEqual(cred.mode, CredentialMode.USER_KEY)

    def test_env_overrides_user_settings(self) -> None:
        set_provider_api_key("openalex", "user-openalex-key")
        os.environ["QUBE_OPENALEX_API_KEY"] = "env-openalex-key"
        cred = resolve_credential("openalex")
        self.assertEqual(cred.secret, "env-openalex-key")
        self.assertEqual(cred.mode, CredentialMode.ENV_KEY)

    def test_anonymous_when_empty(self) -> None:
        cred = resolve_credential("openalex")
        self.assertIsNone(cred.secret)
        self.assertEqual(cred.mode, CredentialMode.ANONYMOUS)

    def test_fixture_mode_ignores_credentials(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"
        set_provider_api_key("openalex", "user-openalex-key")
        os.environ["QUBE_OPENALEX_API_KEY"] = "env-openalex-key"
        cred = resolve_credential("openalex")
        self.assertIsNone(cred.secret)
        self.assertEqual(cred.mode, CredentialMode.FIXTURE)

    def test_credential_bundle_repr_hides_secret(self) -> None:
        bundle = CredentialBundle(
            provider_id="openalex",
            mode=CredentialMode.USER_KEY,
            secret="super-secret",
        )
        text = repr(bundle)
        self.assertIn("openalex", text)
        self.assertNotIn("super-secret", text)
        self.assertIn("***", text)

    def test_clear_provider_api_key(self) -> None:
        set_provider_api_key("ncbi", "ncbi-user-key")
        clear_provider_api_key("ncbi")
        cred = resolve_credential("ncbi")
        self.assertIsNone(cred.secret)
        self.assertEqual(cred.mode, CredentialMode.ANONYMOUS)


class TestCredentialAdapterWiring(unittest.TestCase):
    def setUp(self) -> None:
        self._store = _FakeSettingsStore()
        self._patch_store = patch(
            "core.app_settings._store",
            lambda: self._store,
        )
        self._patch_store.start()

    def tearDown(self) -> None:
        self._patch_store.stop()
        for name in (
            "QUBE_OPENALEX_API_KEY",
            "QUBE_NCBI_API_KEY",
            "QUBE_COURTLISTENER_API_TOKEN",
            "QUBE_KNOWLEDGE_FIXTURES",
        ):
            os.environ.pop(name, None)

    @patch("core.knowledge.adapters.openalex.knowledge_get")
    def test_openalex_uses_user_settings_key(self, mock_get: MagicMock) -> None:
        set_provider_api_key("openalex", "settings-openalex-key")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        mock_get.return_value = resp
        openalex.search_openalex("quantum dots", max_results=1)
        _url, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "settings-openalex-key")

    @patch("core.knowledge.adapters.pubmed_eutils.knowledge_get")
    def test_pubmed_uses_shared_ncbi_user_key(self, mock_get: MagicMock) -> None:
        set_provider_api_key("ncbi", "shared-ncbi-key")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = resp
        pubmed_eutils.search_pubmed("heart failure", max_results=1)
        _url, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "shared-ncbi-key")

    @patch("core.knowledge.adapters.pubchem.knowledge_get")
    def test_pubchem_uses_shared_ncbi_user_key(self, mock_get: MagicMock) -> None:
        set_provider_api_key("ncbi", "shared-ncbi-key")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"IdentifierList": {"CID": []}}
        mock_get.return_value = resp
        with patch(
            "core.knowledge.adapters.pubchem._use_fixtures",
            return_value=False,
        ):
            pubchem.search_pubchem("aspirin", max_results=1)
        self.assertTrue(mock_get.called)
        _url, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "shared-ncbi-key")

    def test_authorization_token_from_settings(self) -> None:
        set_provider_api_key("courtlistener", "cl-user-token")
        self.assertEqual(authorization_token("courtlistener"), "cl-user-token")
        self.assertEqual(resolve("courtlistener").api_key, "cl-user-token")

    def test_merge_query_params_respects_fixture_mode(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"
        set_provider_api_key("openalex", "settings-openalex-key")
        merged = merge_query_params({"search": "test"}, "openalex")
        self.assertEqual(merged["search"], "test")
        self.assertNotIn("api_key", merged)


if __name__ == "__main__":
    unittest.main()
