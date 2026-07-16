"""Tests for discovery Phase 2: privacy tiers, routing, health, SearXNG."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.backoff import reset_discovery_backoff  # noqa: E402
from core.knowledge.discovery.cache import reset_discovery_cache  # noqa: E402
from core.knowledge.discovery.health import (  # noqa: E402
    consume_tier_b_suggestion,
    get_conservative_pacing_multiplier,
    is_conservative_mode_active,
    record_ddg_bot_challenge,
    record_ddg_serp_success,
    reset_discovery_health,
)
from core.knowledge.discovery.pacing import (  # noqa: E402
    effective_discovery_pace_min_seconds,
    reset_discovery_pacing,
)
from core.knowledge.discovery.privacy_policy import (  # noqa: E402
    TIER_BALANCED,
    TIER_PRIVATE,
    TIER_SEARXNG,
    privacy_tier_label,
    resolve_discovery_route,
)
from core.knowledge.discovery.registry import discover_full_with_fallback  # noqa: E402
from core.knowledge.discovery.session_budget import reset_ddg_session_budget  # noqa: E402
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402


class TestPrivacyTierDefaults(unittest.TestCase):
    def setUp(self) -> None:
        from core.settings_store import SettingsStore, reset_settings_store_for_tests
        import core.settings_store as settings_store_module
        import tempfile
        from pathlib import Path

        reset_settings_store_for_tests()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.user_path = Path(self._tmpdir.name) / "settings.json"
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            settings_store_module._store = SettingsStore(user_path=self.user_path)

    def tearDown(self) -> None:
        from core.settings_store import reset_settings_store_for_tests

        reset_settings_store_for_tests()
        self._tmpdir.cleanup()

    def test_schema_default_privacy_tier_is_private(self) -> None:
        from core.settings_store import bundled_settings_schema_path
        import json

        schema = json.loads(bundled_settings_schema_path().read_text(encoding="utf-8"))
        entry = schema["qube.knowledge.discovery_privacy_tier"]
        self.assertEqual(entry["default"], TIER_PRIVATE)
        self.assertEqual(entry["enum"][0], TIER_PRIVATE)

    def test_fresh_settings_use_private_search_label(self) -> None:
        from core.app_settings import get_discovery_privacy_tier

        self.assertEqual(get_discovery_privacy_tier(), TIER_PRIVATE)
        self.assertEqual(
            privacy_tier_label(get_discovery_privacy_tier()),
            "Private search (recommended)",
        )

    def test_searxng_tier_persists_without_instance(self) -> None:
        from core.app_settings import (
            KEY_DISCOVERY_PRIVACY_TIER,
            get_discovery_privacy_tier,
            set_discovery_privacy_tier,
        )
        from core.settings_store import get_settings_store

        set_discovery_privacy_tier(TIER_SEARXNG)
        store = get_settings_store()
        self.assertIn(KEY_DISCOVERY_PRIVACY_TIER, store.all_overrides())
        self.assertEqual(get_discovery_privacy_tier(), TIER_SEARXNG)
        self.assertEqual(
            store.all_overrides()[KEY_DISCOVERY_PRIVACY_TIER],
            TIER_SEARXNG,
        )

    @patch("core.app_settings.get_discovery_searxng_base_url", return_value="")
    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_SEARXNG)
    def test_searxng_tier_without_url_uses_ddg_primary(self, _mock_tier, _mock_url) -> None:
        route = resolve_discovery_route()
        self.assertEqual(route.primary_id, "duckduckgo")
        self.assertIn("wikipedia", route.fallback_ids)


class TestPrivacyTiers(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_backoff()
        reset_discovery_cache()
        reset_discovery_pacing()
        reset_ddg_session_budget()
        reset_discovery_health()

    def tearDown(self) -> None:
        reset_discovery_backoff()
        reset_discovery_cache()
        reset_discovery_pacing()
        reset_ddg_session_budget()
        reset_discovery_health()

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_PRIVATE)
    def test_private_tier_fallback_is_wikipedia_only(self, _mock_tier) -> None:
        route = resolve_discovery_route()
        self.assertEqual(route.primary_id, "duckduckgo")
        self.assertEqual(route.fallback_ids, ("wikipedia",))

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_BALANCED)
    @patch("core.app_settings.get_discovery_api_fallback_enabled", return_value=True)
    @patch("core.knowledge.discovery.privacy_policy.brave_search_configured", return_value=True)
    def test_balanced_tier_includes_brave_fallback(
        self,
        _mock_brave,
        _mock_api,
        _mock_tier,
    ) -> None:
        route = resolve_discovery_route()
        self.assertEqual(route.fallback_ids[0], "brave_search")
        self.assertIn("wikipedia", route.fallback_ids)

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_BALANCED)
    @patch("core.app_settings.get_discovery_api_fallback_enabled", return_value=True)
    @patch("core.knowledge.discovery.privacy_policy.brave_search_configured", return_value=True)
    def test_site_bias_uses_brave_primary_when_configured(
        self,
        _mock_brave,
        _mock_api,
        _mock_tier,
    ) -> None:
        route = resolve_discovery_route(site_bias=("bbcgoodfood.com",))
        self.assertEqual(route.primary_id, "brave_search")
        self.assertTrue(route.site_bias_brave_primary)

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_PRIVATE)
    @patch("core.knowledge.discovery.duckduckgo.DuckDuckGoDiscovery.discover_full")
    def test_private_tier_tags_result(self, mock_ddg, _mock_tier) -> None:
        mock_ddg.return_value = DiscoveryResult(
            candidates=(
                CandidateUrl(
                    url="https://example.org",
                    title="Ex",
                    snippet="Sn",
                    source="duckduckgo",
                ),
            ),
            raw_rows=(),
            search_outcome=SearchOutcome(
                kind=SearchOutcomeKind.SERP_SUCCESS,
                provider="duckduckgo",
                candidate_count=1,
            ),
            provider_id="duckduckgo",
        )
        result = discover_full_with_fallback("birds", max_results=3)
        self.assertEqual(result.privacy_tier, TIER_PRIVATE)


class TestDiscoveryHealth(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_health()

    def tearDown(self) -> None:
        reset_discovery_health()

    def test_conservative_mode_after_two_challenges(self) -> None:
        record_ddg_bot_challenge()
        self.assertFalse(is_conservative_mode_active())
        record_ddg_bot_challenge()
        self.assertTrue(is_conservative_mode_active())
        self.assertGreater(get_conservative_pacing_multiplier(), 1.0)

    def test_success_clears_conservative_mode(self) -> None:
        record_ddg_bot_challenge()
        record_ddg_bot_challenge()
        record_ddg_serp_success()
        self.assertFalse(is_conservative_mode_active())

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_PRIVATE)
    def test_tier_b_suggestion_after_three_challenges(self, _mock_tier) -> None:
        record_ddg_bot_challenge()
        record_ddg_bot_challenge()
        self.assertFalse(consume_tier_b_suggestion())
        record_ddg_bot_challenge()
        self.assertTrue(consume_tier_b_suggestion())
        self.assertFalse(consume_tier_b_suggestion())

    def test_effective_pace_doubles_in_conservative_mode(self) -> None:
        with patch(
            "core.knowledge.discovery.pacing.discovery_pace_min_seconds",
            return_value=3.0,
        ):
            record_ddg_bot_challenge()
            record_ddg_bot_challenge()
            self.assertGreaterEqual(effective_discovery_pace_min_seconds(), 6.0)


class TestSearXNGProvider(unittest.TestCase):
    @patch("core.app_settings.get_discovery_searxng_base_url", return_value="https://search.local")
    @patch("core.knowledge.discovery.searxng.requests.get")
    def test_searxng_parses_json_results(self, mock_get, _mock_url) -> None:
        from core.knowledge.discovery.searxng import SearXNGDiscovery

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.org/page",
                    "content": "Snippet text",
                }
            ]
        }
        mock_get.return_value.raise_for_status = lambda: None

        result = SearXNGDiscovery().discover_full("test", max_results=3)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].url, "https://example.org/page")

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_SEARXNG)
    @patch("core.app_settings.get_discovery_searxng_base_url", return_value="https://search.local")
    def test_searxng_tier_uses_searxng_primary(self, _mock_url, _mock_tier) -> None:
        route = resolve_discovery_route()
        self.assertEqual(route.primary_id, "searxng")


if __name__ == "__main__":
    unittest.main()
