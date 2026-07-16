"""Tests for general web selective fetch pipeline (M5)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.fetch.engine import fetch_html_string  # noqa: E402
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402
from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline  # noqa: E402
from core.knowledge.retrieval_profiles import (  # noqa: E402
    PROFILE_BALANCED,
    PROFILE_FAST,
    get_profile_spec,
)
from core.knowledge.types import RetrievalContext  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "fetch"
_ARTICLE_HTML = (_FIXTURES / "article_clean.html").read_text(encoding="utf-8")
_RECIPE_HTML = (_FIXTURES / "recipe_jsonld.html").read_text(encoding="utf-8")

_CANDIDATES = []


def _discovery_result(*candidates: CandidateUrl) -> DiscoveryResult:
    rows = tuple(
        {
            "title": c.title or "",
            "snippet": c.snippet or "",
            "url": c.url,
        }
        for c in candidates
    )
    return DiscoveryResult(
        candidates=candidates,
        raw_rows=rows,
        search_outcome=SearchOutcome(
            kind=SearchOutcomeKind.SERP_SUCCESS,
            candidate_count=len(candidates),
        ),
    )


class TestGeneralWebFetchPipeline(unittest.TestCase):
    @patch("core.knowledge.pipeline_general_web.discover_full")
    @patch("core.knowledge.pipeline_general_web.fetch_url")
    def test_balanced_profile_fetches_sections(self, mock_fetch, mock_discover_full) -> None:
        mock_discover_full.return_value = _discovery_result(
            CandidateUrl(
                url="https://example.com/birds",
                title="Dust Bathing in Birds",
                snippet="Birds take dust baths.",
                source="duckduckgo",
            )
        )
        mock_fetch.return_value = fetch_html_string(
            _ARTICLE_HTML,
            url="https://example.com/birds",
        )

        balanced = get_profile_spec(PROFILE_BALANCED)
        ctx = RetrievalContext(
            query="Why do birds take dust baths?",
            semantic_query="Why do birds take dust baths?",
            retrieval_profile=balanced.id,
            budget=balanced.budget,
        )
        bundle, rel_diag, _raw = run_general_web_evidence_pipeline(ctx)

        self.assertTrue(bundle.sources)
        self.assertEqual(bundle.retrieval_strategy, "ddg_serp_selective_fetch")
        self.assertTrue(
            all(source.fetch_status == "full_extract" for source in bundle.sources)
        )
        self.assertIsNotNone(rel_diag)
        assert rel_diag is not None
        self.assertEqual(rel_diag.get("fetch_url_count"), 1)
        self.assertIn("fetch_provenance", rel_diag)
        self.assertIn("pipeline_stages", rel_diag)
        mock_fetch.assert_called_once()

    @patch("core.knowledge.pipeline_general_web.discover_full")
    @patch("core.knowledge.pipeline_general_web.fetch_url")
    def test_fast_profile_stays_serp_only(self, mock_fetch, mock_discover_full) -> None:
        mock_discover_full.return_value = _discovery_result(
            CandidateUrl(
                url="https://example.com/birds",
                title="Dust Bathing in Birds",
                snippet="Birds take dust baths.",
                source="duckduckgo",
            )
        )

        fast = get_profile_spec(PROFILE_FAST)
        ctx = RetrievalContext(
            query="Why do birds take dust baths?",
            semantic_query="Why do birds take dust baths?",
            retrieval_profile=fast.id,
            budget=fast.budget,
        )
        bundle, rel_diag, _raw = run_general_web_evidence_pipeline(ctx)

        self.assertTrue(bundle.sources)
        self.assertEqual(bundle.retrieval_strategy, "ddg_serp_relevance_gate")
        self.assertTrue(
            all(source.fetch_status == "snippet_only" for source in bundle.sources)
        )
        mock_fetch.assert_not_called()
        assert rel_diag is not None
        self.assertEqual(rel_diag.get("fetch_url_count"), 0)
        self.assertIn("fetch_provenance", rel_diag)

    @patch("core.knowledge.pipeline_general_web.filter_web_results")
    @patch("core.knowledge.pipeline_general_web.discover_full")
    @patch("core.knowledge.pipeline_general_web.fetch_url")
    def test_recipe_skips_relevance_gate(
        self,
        mock_fetch,
        mock_discover_full,
        mock_filter,
    ) -> None:
        mock_discover_full.return_value = _discovery_result(
            CandidateUrl(
                url="https://seriouseats.com/carbonara",
                title="Authentic Roman Carbonara",
                snippet="A classic pasta dish with guanciale and eggs.",
                source="duckduckgo",
            ),
            CandidateUrl(
                url="https://bbcgoodfood.com/carbonara",
                title="Spaghetti carbonara recipe",
                snippet="Traditional Italian recipe.",
                source="duckduckgo",
            ),
        )
        mock_fetch.return_value = fetch_html_string(
            _RECIPE_HTML,
            url="https://seriouseats.com/carbonara",
        )

        balanced = get_profile_spec(PROFILE_BALANCED)
        ctx = RetrievalContext(
            query="spaghetti carbonara recipe",
            semantic_query="spaghetti carbonara recipe",
            retrieval_profile=balanced.id,
            budget=balanced.budget,
            composer_tool="recipe",
        )
        bundle, rel_diag, _raw = run_general_web_evidence_pipeline(ctx)

        mock_filter.assert_not_called()
        self.assertIsNotNone(rel_diag)
        assert rel_diag is not None
        self.assertTrue(rel_diag.get("web_relevance_gate_skipped"))
        self.assertEqual(rel_diag.get("web_relevance_gate_mode"), "recipe_fetch_skip")
        self.assertTrue(bundle.sources)
        mock_fetch.assert_called_once()

    @patch("core.knowledge.pipeline_general_web.extract_document")
    @patch("core.knowledge.pipeline_general_web.discover_full")
    @patch("core.knowledge.pipeline_general_web.fetch_url")
    def test_extractor_unavailable_falls_back_to_serp_snippets(
        self,
        mock_fetch,
        mock_discover_full,
        mock_extract,
    ) -> None:
        mock_discover_full.return_value = _discovery_result(
            CandidateUrl(
                url="https://example.com/birds",
                title="Dust Bathing in Birds",
                snippet="Birds take dust baths.",
                source="duckduckgo",
            )
        )
        mock_fetch.return_value = fetch_html_string(
            _ARTICLE_HTML,
            url="https://example.com/birds",
        )
        mock_extract.side_effect = RuntimeError(
            "trafilatura is not installed — add it to requirements.txt"
        )

        balanced = get_profile_spec(PROFILE_BALANCED)
        ctx = RetrievalContext(
            query="Why do birds take dust baths?",
            semantic_query="Why do birds take dust baths?",
            retrieval_profile=balanced.id,
            budget=balanced.budget,
            composer_tool="fetch",
        )
        bundle, rel_diag, _raw = run_general_web_evidence_pipeline(ctx)

        self.assertTrue(bundle.sources)
        self.assertEqual(bundle.retrieval_strategy, "ddg_serp_relevance_gate")
        self.assertIn("snippet_fallback", bundle.warnings)
        self.assertTrue(
            all(source.fetch_status == "snippet_only" for source in bundle.sources)
        )
        assert rel_diag is not None
        fetch_diag = rel_diag.get("fetch") or {}
        failed = fetch_diag.get("failed") or []
        self.assertTrue(failed)
        self.assertEqual(failed[0].get("failure_reason"), "extractor_unavailable")


if __name__ == "__main__":
    unittest.main()
