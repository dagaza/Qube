"""Tests for discovery provider backoff and cache."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.backoff import (  # noqa: E402
    consume_ddg_backoff_notification,
    get_provider_backoff,
    is_provider_in_backoff,
    mark_provider_backoff,
    reset_discovery_backoff,
)
from core.knowledge.discovery.pacing import reset_discovery_pacing  # noqa: E402
from core.knowledge.discovery.session_budget import reset_ddg_session_budget  # noqa: E402
from core.knowledge.discovery.cache import (  # noqa: E402
    get_cached_discovery,
    reset_discovery_cache,
    store_cached_discovery,
)
from core.knowledge.discovery.duckduckgo import DuckDuckGoDiscovery  # noqa: E402
from core.knowledge.discovery.registry import discover_full_with_fallback  # noqa: E402
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402


def _ddg_bot_challenge_result() -> DiscoveryResult:
    return DiscoveryResult(
        candidates=(),
        raw_rows=({"title": "", "snippet": "Internet search blocked: DuckDuckGo bot challenge (try again later)."},),
        search_outcome=SearchOutcome(
            kind=SearchOutcomeKind.BOT_CHALLENGE,
            provider="duckduckgo",
            http_status=202,
        ),
        provider_id="duckduckgo",
    )


def _ddg_success_result() -> DiscoveryResult:
    return DiscoveryResult(
        candidates=(
            CandidateUrl(
                url="https://example.org/page",
                title="Example",
                snippet="Snippet",
                source="duckduckgo",
            ),
        ),
        raw_rows=({"title": "Example", "snippet": "Snippet", "url": "https://example.org/page"},),
        search_outcome=SearchOutcome(
            kind=SearchOutcomeKind.SERP_SUCCESS,
            provider="duckduckgo",
            candidate_count=1,
        ),
        provider_id="duckduckgo",
    )


class TestDiscoveryBackoff(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_backoff()
        reset_discovery_cache()
        reset_discovery_pacing()
        reset_ddg_session_budget()

    def tearDown(self) -> None:
        reset_discovery_backoff()
        reset_discovery_cache()
        reset_discovery_pacing()
        reset_ddg_session_budget()

    def test_mark_and_detect_backoff(self) -> None:
        self.assertTrue(mark_provider_backoff("duckduckgo", ttl_seconds=120))
        self.assertTrue(is_provider_in_backoff("duckduckgo"))
        entry = get_provider_backoff("duckduckgo")
        assert entry is not None
        self.assertGreater(entry.remaining_seconds, 0)

    def test_backoff_notification_pending_once(self) -> None:
        mark_provider_backoff("duckduckgo", ttl_seconds=120)
        should_notify, remaining = consume_ddg_backoff_notification()
        self.assertTrue(should_notify)
        self.assertGreater(remaining, 0)
        should_notify_again, _ = consume_ddg_backoff_notification()
        self.assertFalse(should_notify_again)

    def test_repeated_mark_does_not_renotify(self) -> None:
        mark_provider_backoff("duckduckgo", ttl_seconds=120)
        consume_ddg_backoff_notification()
        self.assertFalse(mark_provider_backoff("duckduckgo", ttl_seconds=120))
        should_notify, _ = consume_ddg_backoff_notification()
        self.assertFalse(should_notify)

    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_backoff_skips_ddg_network_call(self, mock_ddg) -> None:
        mark_provider_backoff("duckduckgo", ttl_seconds=600)
        mock_ddg.return_value = _ddg_success_result()

        with patch(
            "core.knowledge.discovery.registry._provider_discover_full"
        ) as mock_provider_call:
            from core.knowledge.discovery.wikipedia import WikipediaDiscovery

            mock_provider_call.side_effect = lambda provider, *args, **kwargs: (
                DiscoveryResult(
                    candidates=(
                        CandidateUrl(
                            url="https://en.wikipedia.org/wiki/Test",
                            title="Test",
                            snippet="Wiki",
                            source="wikipedia",
                        ),
                    ),
                    raw_rows=(),
                    search_outcome=SearchOutcome(
                        kind=SearchOutcomeKind.SERP_SUCCESS,
                        provider="wikipedia",
                        candidate_count=1,
                    ),
                    provider_id="wikipedia",
                )
                if isinstance(provider, WikipediaDiscovery)
                else _ddg_success_result()
            )

            result = discover_full_with_fallback("test query", max_results=3)

        mock_ddg.assert_not_called()
        self.assertEqual(result.provider_id, "wikipedia")

    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_bot_challenge_triggers_backoff(self, mock_ddg) -> None:
        mock_ddg.return_value = _ddg_bot_challenge_result()
        discover_full_with_fallback("blocked query", max_results=3)
        self.assertTrue(is_provider_in_backoff("duckduckgo"))


class TestDiscoveryCache(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_cache()

    def tearDown(self) -> None:
        reset_discovery_cache()

    def test_cache_stores_successful_results(self) -> None:
        result = _ddg_success_result()
        store_cached_discovery(
            "duckduckgo",
            "birds",
            max_results=3,
            site_bias=None,
            result=result,
        )
        cached = get_cached_discovery("duckduckgo", "birds", max_results=3, site_bias=None)
        assert cached is not None
        self.assertEqual(len(cached.candidates), 1)

    def test_cache_ignores_bot_challenge(self) -> None:
        store_cached_discovery(
            "duckduckgo",
            "blocked",
            max_results=3,
            site_bias=None,
            result=_ddg_bot_challenge_result(),
        )
        self.assertIsNone(
            get_cached_discovery("duckduckgo", "blocked", max_results=3, site_bias=None)
        )


if __name__ == "__main__":
    unittest.main()
