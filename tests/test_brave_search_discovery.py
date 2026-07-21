"""Tests for Brave Search discovery provider and fallback chain (M10)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.brave_search import BraveSearchDiscovery  # noqa: E402
from core.knowledge.discovery.policy import (  # noqa: E402
    PRIMARY_DISCOVERY_PROVIDER_ID,
    bot_challenge_fallback_chain,
)
from core.knowledge.discovery.privacy_policy import TIER_BALANCED  # noqa: E402
from core.knowledge.discovery.registry import (  # noqa: E402
    default_discovery_provider,
    discover_full_with_fallback,
)
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402


class TestBraveSearchDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        from core.knowledge.discovery.backoff import reset_discovery_backoff
        from core.knowledge.discovery.cache import reset_discovery_cache

        reset_discovery_backoff()
        reset_discovery_cache()

    def test_default_provider_is_duckduckgo(self) -> None:
        self.assertEqual(default_discovery_provider().id, PRIMARY_DISCOVERY_PROVIDER_ID)

    @patch("core.knowledge.discovery.brave_search.brave_search_configured", return_value=False)
    def test_discover_without_key_returns_no_credentials(self, _mock_cfg) -> None:
        result = BraveSearchDiscovery().discover_full("birds", max_results=3)
        assert result.search_outcome is not None
        self.assertEqual(result.search_outcome.failure_sentinel_reason, "no_credentials")
        self.assertEqual(len(result.candidates), 0)

    @patch("core.knowledge.discovery.brave_search.search_brave")
    @patch("core.knowledge.discovery.brave_search.brave_search_configured", return_value=True)
    def test_discover_maps_rows_to_candidates(self, _mock_cfg, mock_search) -> None:
        mock_search.return_value = (
            [
                {
                    "title": "Birds",
                    "snippet": "Feathered animals",
                    "url": "https://example.org/birds",
                }
            ],
            {"response_kind": "serp", "http_status": 200, "parsed_rows": 1},
        )
        result = BraveSearchDiscovery().discover_full(
            "birds",
            max_results=3,
            site_bias=("example.org",),
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].url, "https://example.org/birds")
        assert result.search_outcome is not None
        self.assertEqual(result.search_outcome.kind, SearchOutcomeKind.SERP_SUCCESS)

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_BALANCED)
    @patch("core.app_settings.get_discovery_api_fallback_enabled", return_value=True)
    @patch("core.knowledge.discovery.privacy_policy.brave_search_configured", return_value=True)
    def test_fallback_chain_includes_brave_when_configured(
        self,
        _mock_brave,
        _mock_api,
        _mock_tier,
    ) -> None:
        chain = bot_challenge_fallback_chain()
        self.assertEqual(chain[0], "brave_search")
        self.assertIn("wikipedia", chain)

    @patch("core.app_settings.get_discovery_privacy_tier", return_value=TIER_BALANCED)
    @patch("core.app_settings.get_discovery_api_fallback_enabled", return_value=True)
    @patch("core.knowledge.discovery.privacy_policy.brave_search_configured", return_value=True)
    @patch.object(BraveSearchDiscovery, "discover_full")
    def test_site_bias_uses_brave_primary_when_configured(
        self,
        mock_brave,
        _mock_brave_cfg,
        _mock_api,
        _mock_tier,
    ) -> None:
        mock_brave.return_value = DiscoveryResult(
            candidates=(
                CandidateUrl(
                    url="https://seriouseats.com/carbonara",
                    title="Carbonara",
                    snippet="Classic pasta",
                    source="brave_search",
                ),
            ),
            raw_rows=({"title": "Carbonara", "snippet": "Classic pasta", "url": "https://seriouseats.com/carbonara"},),
            search_outcome=SearchOutcome(
                kind=SearchOutcomeKind.SERP_SUCCESS,
                provider="brave_search",
                candidate_count=1,
            ),
            provider_id="brave_search",
        )

        result = discover_full_with_fallback(
            "carbonara recipe",
            max_results=3,
            site_bias=("seriouseats.com",),
        )

        self.assertEqual(result.provider_id, "brave_search")
        mock_brave.assert_called_once()
        call_kwargs = mock_brave.call_args.kwargs
        self.assertEqual(call_kwargs.get("site_bias"), ("seriouseats.com",))


if __name__ == "__main__":
    unittest.main()
