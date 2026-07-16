"""Tests for discovery Phase 1: pacing, session budget, cache normalization."""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.cache import (  # noqa: E402
    discovery_cache_ttl_for_profile,
    get_cached_discovery,
    reset_discovery_cache,
    store_cached_discovery,
)
from core.knowledge.discovery.pacing import (  # noqa: E402
    discovery_pace_min_seconds,
    reset_discovery_pacing,
    wait_for_ddg_pace_slot,
)
from core.knowledge.discovery.query_normalization import (  # noqa: E402
    normalize_discovery_query,
)
from core.knowledge.discovery.registry import discover_full_with_fallback  # noqa: E402
from core.knowledge.discovery.session_budget import (  # noqa: E402
    get_ddg_budget_block_reason,
    get_ddg_burst_budget_status,
    get_ddg_session_budget_status,
    is_ddg_burst_budget_exhausted,
    is_ddg_session_budget_exhausted,
    record_ddg_live_request,
    reset_ddg_session_budget,
)
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.discovery.duckduckgo import DuckDuckGoDiscovery  # noqa: E402
from core.knowledge.discovery.backoff import reset_discovery_backoff  # noqa: E402
from core.knowledge.pipeline_trusted import TrustedEvidencePipeline  # noqa: E402
from core.knowledge.retrieval_profiles import PROFILE_BALANCED, PROFILE_FAST  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_TRUSTED_KNOWLEDGE  # noqa: E402


def _ddg_success_result() -> DiscoveryResult:
    return DiscoveryResult(
        candidates=(
            CandidateUrl(
                url="https://www.gov.uk/example",
                title="Example",
                snippet="Snippet",
                source="duckduckgo",
            ),
        ),
        raw_rows=({"title": "Example", "snippet": "Snippet", "url": "https://www.gov.uk/example"},),
        search_outcome=SearchOutcome(
            kind=SearchOutcomeKind.SERP_SUCCESS,
            provider="duckduckgo",
            candidate_count=1,
        ),
        provider_id="duckduckgo",
    )


class TestQueryNormalization(unittest.TestCase):
    def test_normalizes_case_whitespace_and_trailing_question(self) -> None:
        self.assertEqual(
            normalize_discovery_query("  What Are Birds?  "),
            "what are birds",
        )


class TestDiscoveryPacing(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_pacing()

    def tearDown(self) -> None:
        reset_discovery_pacing()

    def test_second_slot_waits_at_least_min_interval(self) -> None:
        acquired1, _ = wait_for_ddg_pace_slot(max_wait_sec=5.0)
        self.assertTrue(acquired1)
        started = time.time()
        acquired2, wait_ms = wait_for_ddg_pace_slot(max_wait_sec=10.0)
        elapsed = time.time() - started
        self.assertTrue(acquired2)
        self.assertGreaterEqual(elapsed, discovery_pace_min_seconds() * 0.9)
        self.assertGreaterEqual(wait_ms, int(discovery_pace_min_seconds() * 500))

    def test_pacing_timeout_when_max_wait_too_short(self) -> None:
        wait_for_ddg_pace_slot(max_wait_sec=5.0)
        acquired, wait_ms = wait_for_ddg_pace_slot(max_wait_sec=0.1)
        self.assertFalse(acquired)
        self.assertGreaterEqual(wait_ms, 0)


class TestBurstBudget(unittest.TestCase):
    def setUp(self) -> None:
        reset_ddg_session_budget()

    def tearDown(self) -> None:
        reset_ddg_session_budget()

    @patch.dict(
        os.environ,
        {
            "QUBE_DDG_BURST_BUDGET": "2",
            "QUBE_DDG_SESSION_BUDGET": "10",
        },
    )
    def test_burst_exhausts_before_session(self) -> None:
        self.assertFalse(is_ddg_burst_budget_exhausted())
        record_ddg_live_request()
        record_ddg_live_request()
        self.assertTrue(is_ddg_burst_budget_exhausted())
        self.assertFalse(is_ddg_session_budget_exhausted())
        self.assertEqual(get_ddg_budget_block_reason(), "burst")
        burst = get_ddg_burst_budget_status()
        self.assertEqual(burst.used, 2)
        self.assertEqual(burst.limit, 2)

    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_registry_skips_ddg_when_burst_exhausted(self, mock_ddg) -> None:
        with patch.dict(os.environ, {"QUBE_DDG_BURST_BUDGET": "1", "QUBE_DDG_SESSION_BUDGET": "10"}):
            reset_ddg_session_budget()
            record_ddg_live_request()
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
                result = discover_full_with_fallback("burst test", max_results=3)

        mock_ddg.assert_not_called()
        self.assertEqual(result.provider_id, "wikipedia")


class TestSessionBudget(unittest.TestCase):
    def setUp(self) -> None:
        reset_ddg_session_budget()

    def tearDown(self) -> None:
        reset_ddg_session_budget()

    @patch.dict(os.environ, {"QUBE_DDG_SESSION_BUDGET": "2"})
    def test_budget_exhausts_after_live_requests(self) -> None:
        self.assertFalse(is_ddg_session_budget_exhausted())
        record_ddg_live_request()
        record_ddg_live_request()
        self.assertTrue(is_ddg_session_budget_exhausted())
        status = get_ddg_session_budget_status()
        self.assertEqual(status.used, 2)
        self.assertEqual(status.limit, 2)

    @patch.object(DuckDuckGoDiscovery, "discover_full")
    def test_registry_skips_ddg_when_budget_exhausted(self, mock_ddg) -> None:
        with patch.dict(os.environ, {"QUBE_DDG_SESSION_BUDGET": "1"}):
            reset_ddg_session_budget()
            record_ddg_live_request()
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
                result = discover_full_with_fallback("budget test", max_results=3)

        mock_ddg.assert_not_called()
        self.assertEqual(result.provider_id, "wikipedia")


class TestProfileAwareCache(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_cache()

    def tearDown(self) -> None:
        reset_discovery_cache()

    def test_fast_profile_uses_longer_ttl_than_balanced(self) -> None:
        self.assertGreater(
            discovery_cache_ttl_for_profile(PROFILE_FAST),
            discovery_cache_ttl_for_profile(PROFILE_BALANCED),
        )

    def test_normalized_queries_share_cache_key(self) -> None:
        result = _ddg_success_result()
        store_cached_discovery(
            "duckduckgo",
            "What are Birds?",
            max_results=3,
            site_bias=None,
            result=result,
            retrieval_profile=PROFILE_BALANCED,
        )
        cached = get_cached_discovery(
            "duckduckgo",
            "what are birds",
            max_results=3,
            site_bias=None,
            retrieval_profile=PROFILE_BALANCED,
        )
        assert cached is not None
        self.assertTrue(cached.discovery_cache_hit)


class TestTrustedPipelineRegistry(unittest.TestCase):
    def setUp(self) -> None:
        reset_discovery_backoff()
        reset_discovery_cache()
        reset_ddg_session_budget()

    def tearDown(self) -> None:
        reset_discovery_backoff()
        reset_discovery_cache()
        reset_ddg_session_budget()

    @patch("core.knowledge.pipeline_trusted.discover_full_with_fallback")
    @patch("core.knowledge.pipeline_trusted.search_wikipedia")
    def test_trusted_uses_discovery_registry(self, mock_wiki, mock_discover) -> None:
        mock_wiki.return_value = []
        mock_discover.return_value = _ddg_success_result()

        ctx = RetrievalContext(
            query="gov policy",
            semantic_query="gov policy",
            knowledge_service=SERVICE_TRUSTED_KNOWLEDGE,
            budget=RetrievalBudget(max_results=3),
        )
        pipeline = TrustedEvidencePipeline()
        bundle, rel_diag, _raw = pipeline.run(ctx)

        mock_discover.assert_called_once()
        self.assertIsNotNone(bundle)
        assert rel_diag is not None
        self.assertEqual(rel_diag.get("discovery_provider"), "duckduckgo")


if __name__ == "__main__":
    unittest.main()
