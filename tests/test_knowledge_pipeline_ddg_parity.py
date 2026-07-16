"""Parity tests: legacy web retrieval vs v2 evidence pipeline."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters.duckduckgo import is_failure_sentinel  # noqa: E402
from core.knowledge.web_retrieval import (  # noqa: E402
    run_legacy_web_retrieval,
    run_v2_web_retrieval,
)


_SAMPLE_ROWS = [
    {
        "title": "Dust bathing in birds",
        "snippet": "Many bird species take dust baths to clean feathers.",
        "url": "https://example.org/birds",
    },
    {
        "title": "Online search tips",
        "snippet": "How to search the web for answers.",
    },
]


class TestKnowledgePipelineDdgParity(unittest.TestCase):
    def setUp(self) -> None:
        from core.knowledge.discovery.pacing import reset_discovery_pacing
        from core.knowledge.discovery.session_budget import reset_ddg_session_budget

        reset_discovery_pacing()
        reset_ddg_session_budget()

    @patch("core.knowledge.discovery.duckduckgo.search_duckduckgo_detailed")
    def test_legacy_and_v2_keep_same_rows(self, mock_detailed) -> None:
        mock_detailed.return_value = (
            [dict(r) for r in _SAMPLE_ROWS],
            {
                "response_kind": "serp",
                "http_status": 200,
                "parsed_rows": len(_SAMPLE_ROWS),
                "bot_challenge_signals": [],
                "pace_wait_ms": 0,
            },
        )
        query = "Why do birds take dust baths?"

        legacy = run_legacy_web_retrieval(
            query=query,
            semantic_query=query,
            embed_fn=None,
            query_vector=None,
        )
        v2 = run_v2_web_retrieval(
            query=query,
            semantic_query=query,
            embed_fn=None,
            query_vector=None,
        )

        self.assertFalse(legacy.skip_enrichment)
        self.assertFalse(v2.skip_enrichment)
        self.assertIsNotNone(legacy.web_results)
        self.assertIsNotNone(v2.web_results)
        assert legacy.web_results is not None
        assert v2.web_results is not None
        self.assertEqual(
            [r.get("title") for r in legacy.web_results],
            [r.get("title") for r in v2.web_results],
        )
        self.assertIsNotNone(v2.bundle)
        assert v2.bundle is not None
        self.assertEqual(len(v2.bundle.sources), len(legacy.web_results))

    @patch("core.knowledge.discovery.duckduckgo.search_duckduckgo_detailed")
    def test_failure_sentinel_skips_enrichment(self, mock_detailed) -> None:
        mock_detailed.return_value = (
            [{"title": "", "snippet": "No relevant internet results found."}],
            {
                "response_kind": "empty_parse",
                "http_status": 200,
                "parsed_rows": 0,
                "bot_challenge_signals": [],
                "pace_wait_ms": 0,
            },
        )
        legacy = run_legacy_web_retrieval(query="q", semantic_query="q")
        v2 = run_v2_web_retrieval(query="q", semantic_query="q")
        self.assertTrue(legacy.skip_enrichment)
        self.assertTrue(v2.skip_enrichment)
        self.assertIsNone(legacy.web_results)
        self.assertIsNone(v2.web_results)

    def test_is_failure_sentinel(self) -> None:
        self.assertTrue(
            is_failure_sentinel(
                [{"title": "", "snippet": "Internet search failed due to network error: x"}]
            )
        )
        self.assertFalse(
            is_failure_sentinel([{"title": "Ok", "snippet": "Real content here."}])
        )


if __name__ == "__main__":
    unittest.main()
