"""Tests for live source access badge derivation."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters.catalog import AdapterCatalogEntry, get_adapter_entry  # noqa: E402
from core.knowledge.credentials import set_provider_api_key  # noqa: E402
from core.knowledge.source_access_summary import summarize_source_access  # noqa: E402


class _FakeSettingsStore:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value


class TestSourceAccessSummary(unittest.TestCase):
    def setUp(self) -> None:
        self._store = _FakeSettingsStore()
        self._patch_store = patch(
            "core.app_settings._store",
            lambda: self._store,
        )
        self._patch_store.start()

    def tearDown(self) -> None:
        self._patch_store.stop()

    def test_crossref_is_free_without_configure(self) -> None:
        entry = get_adapter_entry("crossref")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertEqual(summary.badge, "free")
        self.assertFalse(summary.configure_available)
        self.assertIsNone(summary.provider_id)

    def test_openalex_anonymous_optional_key_with_limit_hint(self) -> None:
        entry = get_adapter_entry("openalex")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertEqual(summary.badge, "optional_key")
        self.assertTrue(summary.configure_available)
        self.assertEqual(summary.provider_id, "openalex")
        self.assertEqual(summary.hint, "~$0.10/day")

    def test_openalex_with_user_key_connected(self) -> None:
        set_provider_api_key("openalex", "test-key")
        entry = get_adapter_entry("openalex")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertEqual(summary.badge, "connected")
        self.assertEqual(summary.badge_label, "Connected")

    def test_semantic_scholar_key_required(self) -> None:
        entry = get_adapter_entry("semantic_scholar")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertEqual(summary.badge, "key_required")
        self.assertTrue(summary.configure_available)

    def test_pubchem_shares_ncbi_key_hint(self) -> None:
        entry = get_adapter_entry("pubchem")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertEqual(summary.provider_id, "ncbi")
        self.assertEqual(summary.hint, "Same key as PubMed")

    def test_pubmed_ncbi_primary_has_limit_not_sibling_hint(self) -> None:
        entry = get_adapter_entry("pubmed")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertEqual(summary.provider_id, "ncbi")
        self.assertEqual(summary.hint, "3 req/sec")

    def test_unimplemented_keyed_source_coming_soon(self) -> None:
        entry = AdapterCatalogEntry(
            "future_source",
            "Future Source",
            "scientific_evidence",
            "Science",
            implemented=False,
            requires_api_key=True,
        )
        summary = summarize_source_access(entry)
        self.assertEqual(summary.badge, "coming_soon")
        self.assertTrue(summary.needs_setup)
        self.assertFalse(summary.configure_available)

    def test_crossref_does_not_need_setup(self) -> None:
        entry = get_adapter_entry("crossref")
        assert entry is not None
        summary = summarize_source_access(entry)
        self.assertFalse(summary.needs_setup)


if __name__ == "__main__":
    unittest.main()
