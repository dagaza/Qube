"""Tests for fetch provenance trail (M7)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.search_outcome import (  # noqa: E402
    SearchOutcome,
    SearchOutcomeKind,
)
from core.knowledge.fetch.engine import fetch_html_string  # noqa: E402
from core.knowledge.fetch_provenance import (  # noqa: E402
    FetchProvenance,
    build_fetch_provenance,
    build_pipeline_stages_from_provenance,
    fetch_provenance_from_relevance_diag,
    format_fetch_provenance_text,
    format_pipeline_stages_summary,
    summarize_web_pipeline_outcome,
)
from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace  # noqa: E402
from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline  # noqa: E402
from core.knowledge.retrieval_profiles import PROFILE_BALANCED, get_profile_spec  # noqa: E402
from core.knowledge.types import RetrievalContext  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "fetch"
_RECIPE_HTML = (_FIXTURES / "recipe_jsonld.html").read_text(encoding="utf-8")


class TestFetchProvenance(unittest.TestCase):
    def test_round_trip_dict(self) -> None:
        original = FetchProvenance(
            query="carbonara recipe",
            composer_tool="recipe",
            site_bias=("seriouseats.com", "bbcgoodfood.com"),
            discovery_provider="duckduckgo",
            candidates=(
                {"url": "https://example.com/a", "rank": 0, "title": "A"},
            ),
            selected_urls=("https://example.com/a",),
            fetch_attempts=(
                {
                    "url": "https://example.com/a",
                    "tier": "http",
                    "success": True,
                    "total_bytes": 1200,
                },
            ),
            extractor_name="RecipeExtractor",
            extractor_version="1.0.0",
            extractor_confidence=0.98,
            sections_emitted=1,
            fetch_url_count=1,
            structured_data_type="Recipe",
            document_sections=2,
        )
        restored = FetchProvenance.from_dict(original.to_dict())
        assert restored is not None
        self.assertEqual(restored.query, original.query)
        self.assertEqual(restored.extractor_name, "RecipeExtractor")
        self.assertEqual(restored.structured_data_type, "Recipe")

    def test_format_fetch_provenance_text_recipe_chain(self) -> None:
        provenance = FetchProvenance(
            query="carbonara recipe",
            composer_tool="recipe",
            site_bias=("seriouseats.com", "bbcgoodfood.com"),
            discovery_provider="duckduckgo",
            candidates=(
                {"url": "https://www.bbcgoodfood.com/recipes/carbonara", "rank": 0, "title": "Carbonara"},
            ),
            selected_urls=("https://www.bbcgoodfood.com/recipes/carbonara",),
            fetch_attempts=(
                {
                    "url": "https://www.bbcgoodfood.com/recipes/carbonara",
                    "tier": "http",
                    "success": True,
                    "total_bytes": 48231,
                },
            ),
            extractor_name="RecipeExtractor",
            extractor_version="1.0.0",
            extractor_confidence=0.98,
            sections_emitted=1,
            fetch_url_count=1,
            structured_data_type="Recipe",
            document_sections=2,
        )
        text = format_fetch_provenance_text(provenance)
        self.assertIn('Query: "carbonara recipe"', text)
        self.assertIn("Composer: @[tool:recipe]", text)
        self.assertIn("site_bias: [seriouseats.com, bbcgoodfood.com]", text)
        self.assertIn("RecipeExtractor", text)
        self.assertIn("structured_data: Recipe", text)
        self.assertIn("bytes: 48231", text)

    def test_pipeline_stages_include_fetch_and_extract(self) -> None:
        provenance = build_fetch_provenance(
            query="birds",
            composer_tool=None,
            site_bias=None,
            discovery_provider="duckduckgo",
            candidates=[
                CandidateUrl(
                    url="https://example.com/birds",
                    title="Birds",
                    snippet="Dust baths",
                    source="duckduckgo",
                )
            ],
            selected_urls=["https://example.com/birds"],
            fetch_diag={
                "attempts": [
                    {
                        "url": "https://example.com/birds",
                        "tier": "http",
                        "success": True,
                        "total_bytes": 900,
                    }
                ],
                "succeeded": [
                    {
                        "extractor": "TrafilaturaExtractor",
                        "extractor_version": "1.0.0",
                        "extractor_confidence": 0.3,
                        "document_sections": 3,
                    }
                ],
            },
            sections_emitted=2,
            fetch_url_count=1,
        )
        stages = build_pipeline_stages_from_provenance(provenance, rejected_count=1, latency_ms=88.0)
        stage_names = [stage["stage"] for stage in stages]
        self.assertIn("discovery", stage_names)
        self.assertIn("fetch", stage_names)
        self.assertIn("extract", stage_names)
        self.assertIn("section_rank", stage_names)

    @patch("core.knowledge.pipeline_general_web.discover_full")
    @patch("core.knowledge.pipeline_general_web.fetch_url")
    def test_pipeline_attaches_fetch_provenance_to_rel_diag(
        self,
        mock_fetch,
        mock_discover_full,
    ) -> None:
        candidate = CandidateUrl(
            url="https://example.com/recipes/carbonara",
            title="Carbonara",
            snippet="Classic pasta",
            source="duckduckgo",
        )
        mock_discover_full.return_value = DiscoveryResult(
            candidates=(candidate,),
            raw_rows=(
                {
                    "title": "Carbonara",
                    "snippet": "Classic pasta",
                    "url": "https://example.com/recipes/carbonara",
                },
            ),
            search_outcome=SearchOutcome(
                kind=SearchOutcomeKind.SERP_SUCCESS,
                candidate_count=1,
            ),
        )
        mock_fetch.return_value = fetch_html_string(
            _RECIPE_HTML,
            url="https://example.com/recipes/carbonara",
        )

        balanced = get_profile_spec(PROFILE_BALANCED)
        ctx = RetrievalContext(
            query="carbonara recipe",
            semantic_query="carbonara recipe",
            retrieval_profile=balanced.id,
            budget=balanced.budget,
            composer_tool="recipe",
        )
        bundle, rel_diag, _raw = run_general_web_evidence_pipeline(ctx)

        self.assertIsNotNone(rel_diag)
        assert rel_diag is not None
        provenance = fetch_provenance_from_relevance_diag(rel_diag)
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(provenance.composer_tool, "recipe")
        self.assertGreater(provenance.fetch_url_count, 0)
        self.assertEqual(provenance.extractor_name, "RecipeExtractor")
        self.assertIn("pipeline_stages", rel_diag)
        self.assertTrue(bundle.sources)

        trace = build_retrieval_trace(
            bundle,
            relevance_diag=rel_diag,
            retrieval_profile=balanced.id,
            pipeline_stages=rel_diag.get("pipeline_stages"),
        )
        payload = serialize_retrieval_trace(trace, sources=bundle.sources)
        self.assertIn("fetch_provenance", payload["relevance_diag"])
        self.assertTrue(payload.get("pipeline_stages"))
        self.assertEqual(
            payload["relevance_diag"]["search_outcome"]["kind"],
            "serp_success",
        )

    def test_format_pipeline_stages_summary(self) -> None:
        stages = [
            {"stage": "discovery", "outputs_count": 3, "site_bias": ["seriouseats.com"]},
            {"stage": "relevance_gate", "outputs_count": 2, "rejected_count": 1},
            {"stage": "fetch", "fetch_url_count": 1, "outputs_count": 1},
            {
                "stage": "extract",
                "adapter": "TrafilaturaExtractor",
                "outputs_count": 5,
            },
            {"stage": "section_rank", "outputs_count": 3},
            {"stage": "bundle", "latency_ms": 842.5},
        ]
        summary = format_pipeline_stages_summary(stages)
        self.assertIn("discovery(3 urls, site_bias)", summary)
        self.assertIn("relevance_gate(kept=2, dropped=1)", summary)
        self.assertIn("fetch(count=1)", summary)
        self.assertIn("extract(TrafilaturaExtractor, 5 sections)", summary)
        self.assertIn("section_rank(3 ranked)", summary)
        self.assertIn("bundle(842.5ms)", summary)

    def test_summarize_web_pipeline_outcome_snippet_only(self) -> None:
        from core.knowledge.bundle_builder import build_general_web_bundle

        bundle = build_general_web_bundle(
            query_raw="birds",
            query_resolved="birds",
            kept_rows=[
                {"title": "Birds", "snippet": "Dust bathing behavior", "url": "https://x.test"}
            ],
            rejected_count=0,
            latency_ms=120.0,
        )
        rel_diag = {
            "fetch_provenance": {"fetch_url_count": 1},
            "pipeline_stages": [
                {"stage": "discovery", "outputs_count": 3},
                {"stage": "relevance_gate", "outputs_count": 1, "rejected_count": 2},
                {"stage": "fetch", "fetch_url_count": 1, "outputs_count": 1},
                {"stage": "bundle", "latency_ms": 120.0},
            ],
        }
        summary = summarize_web_pipeline_outcome(bundle, rel_diag)
        self.assertEqual(summary["strategy"], "ddg_serp_relevance_gate")
        self.assertEqual(summary["fetch_url_count"], 1)
        self.assertEqual(summary["source_count"], 1)
        self.assertIsNone(summary["search_outcome_kind"])
        self.assertIn("fetch(count=1)", summary["stages_summary"])

    def test_summarize_web_pipeline_outcome_includes_search_outcome_kind(self) -> None:
        from core.knowledge.bundle_builder import build_empty_bundle

        bundle = build_empty_bundle(
            query_raw="birds",
            query_resolved="birds",
            latency_ms=50.0,
            stop_reason="failure_sentinel",
        )
        rel_diag = {
            "search_outcome": {
                "kind": "bot_challenge",
                "provider": "duckduckgo",
                "http_status": 202,
            }
        }
        summary = summarize_web_pipeline_outcome(bundle, rel_diag)
        self.assertEqual(summary["search_outcome_kind"], "bot_challenge")


if __name__ == "__main__":
    unittest.main()
