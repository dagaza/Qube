"""Tests for evidence bundle assembly (Phase 0)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.bundle_builder import (  # noqa: E402
    build_empty_bundle,
    build_general_web_bundle,
)
from core.knowledge.types import COVERAGE_NONE, COVERAGE_POOR  # noqa: E402
from core.knowledge.ui_adapter import (  # noqa: E402
    bundle_to_prompt_context,
    bundle_to_ui_sources,
)


class TestEvidenceBundle(unittest.TestCase):
    def test_empty_bundle(self) -> None:
        bundle = build_empty_bundle(
            query_raw="test query",
            query_resolved="test query",
            latency_ms=12.5,
            stop_reason="failure_sentinel",
        )
        self.assertEqual(bundle.coverage, COVERAGE_NONE)
        self.assertEqual(bundle.confidence, 0.0)
        self.assertEqual(len(bundle.sources), 0)
        summary = bundle.summary_for_skills()
        self.assertFalse(summary.present)
        self.assertEqual(summary.source_count, 0)

    def test_general_web_bundle_from_rows(self) -> None:
        rows = [
            {
                "title": "Dust bathing in birds",
                "snippet": "Birds take dust baths to remove parasites.",
                "url": "https://example.org/birds",
                "_web_token_overlap": 0.42,
            },
            {
                "title": "Avian hygiene",
                "snippet": "Many species dust-bathe regularly.",
                "_web_token_overlap": 0.31,
            },
        ]
        bundle = build_general_web_bundle(
            query_raw="Why do birds take dust baths?",
            query_resolved="Why do birds take dust baths?",
            kept_rows=rows,
            rejected_count=1,
            latency_ms=88.0,
        )
        self.assertEqual(len(bundle.sources), 2)
        self.assertGreater(bundle.confidence, 0.0)
        self.assertEqual(bundle.knowledge_service, "general_web")
        self.assertIn("serp_snippet_only", bundle.warnings)

    def test_ui_adapter_legacy_shape(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[
                {
                    "title": "Example",
                    "snippet": "Body text",
                    "url": "https://example.com",
                    "_web_token_overlap": 0.5,
                }
            ],
            rejected_count=0,
            latency_ms=1.0,
        )
        ui = bundle_to_ui_sources(bundle)
        self.assertEqual(len(ui), 1)
        self.assertEqual(ui[0]["type"], "web")
        self.assertEqual(ui[0]["filename"], "Example")
        self.assertEqual(ui[0]["content"], "Body text")
        self.assertEqual(ui[0]["url"], "https://example.com")
        self.assertEqual(ui[0]["evidence_id"], "ek_1")
        self.assertEqual(ui[0]["source_adapter"], "duckduckgo")

    def test_prompt_context_includes_header(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[{"title": "T", "snippet": "S"}],
            rejected_count=0,
            latency_ms=1.0,
        )
        text = bundle_to_prompt_context(bundle, char_budget=2000)
        self.assertIn("WEB SEARCH RESULTS", text)
        self.assertIn("T", text)
        self.assertIn("S", text)

    def test_single_source_coverage_poor(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[{"title": "Only", "snippet": "One hit"}],
            rejected_count=2,
            latency_ms=1.0,
        )
        self.assertEqual(bundle.coverage, COVERAGE_POOR)


if __name__ == "__main__":
    unittest.main()
