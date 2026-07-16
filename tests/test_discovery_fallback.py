"""Tests for discovery fallback on DuckDuckGo bot_challenge (M9)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.duckduckgo import DuckDuckGoDiscovery  # noqa: E402
from core.knowledge.discovery.registry import (  # noqa: E402
    discover_full,
    discover_full_with_fallback,
    list_discovery_providers,
)
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.discovery.wikipedia import WikipediaDiscovery  # noqa: E402
from core.knowledge.search_outcome import (  # noqa: E402
    SearchOutcome,
    SearchOutcomeKind,
)


def _ddg_bot_challenge_result() -> DiscoveryResult:
    return DiscoveryResult(
        candidates=(),
        raw_rows=(
            {
                "title": "",
                "snippet": (
                    "Internet search blocked: DuckDuckGo bot challenge (try again later)."
                ),
            },
        ),
        search_outcome=SearchOutcome(
            kind=SearchOutcomeKind.BOT_CHALLENGE,
            provider="duckduckgo",
            http_status=202,
            bot_challenge_signals=("http_202", "no_serp_markers"),
            failure_sentinel_reason="ddg_bot_challenge",
        ),
        provider_id="duckduckgo",
    )


def _wiki_success_result() -> DiscoveryResult:
    candidate = CandidateUrl(
        url="https://en.wikipedia.org/wiki/Dust_bathing",
        title="Dust bathing",
        snippet="Birds take dust baths as part of grooming.",
        source="wikipedia",
    )
    return DiscoveryResult(
        candidates=(candidate,),
        raw_rows=(
            {
                "title": "Dust bathing",
                "snippet": "Birds take dust baths as part of grooming.",
                "url": "https://en.wikipedia.org/wiki/Dust_bathing",
            },
        ),
        search_outcome=SearchOutcome(
            kind=SearchOutcomeKind.SERP_SUCCESS,
            provider="wikipedia",
            parsed_rows=1,
            candidate_count=1,
        ),
        provider_id="wikipedia",
    )


class TestDiscoveryFallback(unittest.TestCase):
    def setUp(self) -> None:
        from core.knowledge.discovery.backoff import reset_discovery_backoff
        from core.knowledge.discovery.cache import reset_discovery_cache

        reset_discovery_backoff()
        reset_discovery_cache()

    def test_registry_lists_wikipedia(self) -> None:
        providers = list_discovery_providers()
        self.assertIn("duckduckgo", providers)
        self.assertIn("wikipedia", providers)
        self.assertIn("brave_search", providers)

    @patch.object(WikipediaDiscovery, "discover_full")
    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_fallback_on_bot_challenge(self, mock_ddg, mock_wiki) -> None:
        mock_ddg.return_value = _ddg_bot_challenge_result()
        mock_wiki.return_value = _wiki_success_result()

        result = discover_full_with_fallback("dust baths", max_results=3)

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.provider_id, "wikipedia")
        assert result.search_outcome is not None
        self.assertEqual(result.search_outcome.kind, SearchOutcomeKind.SERP_SUCCESS)
        self.assertEqual(result.search_outcome.fallback_from, "duckduckgo")
        self.assertEqual(result.search_outcome.fallback_reason, "bot_challenge")
        mock_wiki.assert_called_once_with(
            "dust baths",
            max_results=3,
            site_bias=None,
        )

    @patch.object(WikipediaDiscovery, "discover_full")
    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_no_fallback_when_ddg_succeeds(self, mock_ddg, mock_wiki) -> None:
        mock_ddg.return_value = DiscoveryResult(
            candidates=(
                CandidateUrl(
                    url="https://example.org/birds",
                    title="Birds",
                    snippet="Dust bathing",
                    source="duckduckgo",
                ),
            ),
            raw_rows=({"title": "Birds", "snippet": "Dust bathing", "url": "https://example.org/birds"},),
            search_outcome=SearchOutcome(
                kind=SearchOutcomeKind.SERP_SUCCESS,
                provider="duckduckgo",
                candidate_count=1,
            ),
            provider_id="duckduckgo",
        )

        result = discover_full("birds", max_results=3)

        self.assertEqual(result.provider_id, "duckduckgo")
        mock_wiki.assert_not_called()

    @patch.object(WikipediaDiscovery, "discover_full")
    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_fallback_keeps_primary_when_secondary_empty(
        self,
        mock_ddg,
        mock_wiki,
    ) -> None:
        mock_ddg.return_value = _ddg_bot_challenge_result()
        mock_wiki.return_value = DiscoveryResult(
            candidates=(),
            raw_rows=(),
            search_outcome=SearchOutcome(
                kind=SearchOutcomeKind.NO_RESULTS,
                provider="wikipedia",
            ),
            provider_id="wikipedia",
        )

        result = discover_full_with_fallback("obscure topic", max_results=3)

        self.assertEqual(result.provider_id, "duckduckgo")
        assert result.search_outcome is not None
        self.assertEqual(result.search_outcome.kind, SearchOutcomeKind.BOT_CHALLENGE)
        self.assertIsNone(result.search_outcome.fallback_from)

    @patch.object(WikipediaDiscovery, "discover_full")
    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_explicit_provider_skips_fallback(self, mock_ddg, mock_wiki) -> None:
        mock_ddg.return_value = _ddg_bot_challenge_result()

        result = discover_full("dust baths", max_results=3, provider_id="duckduckgo")

        self.assertEqual(result.provider_id, "duckduckgo")
        mock_wiki.assert_not_called()


if __name__ == "__main__":
    unittest.main()
