"""Tests for deep-research LLM synthesis (Phase 4 slice 3)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research_synthesis import (  # noqa: E402
    _strip_redundant_findings_heading,
    build_numbered_retrieval_context,
    compose_deep_research_report,
    synthesize_deep_research_findings,
)
from core.knowledge.types import EvidenceObject, EvidenceBundle  # noqa: E402


def _bundle(*, sources: tuple[EvidenceObject, ...], query: str = "test") -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1",
        query_raw=query,
        query_resolved=query,
        knowledge_service="scientific_evidence",
        retrieval_strategy="deep_research_merged",
        profile_version="0.1.0",
        retrieved_at=0.0,
        latency_ms=10.0,
        confidence=0.85,
        coverage="adequate",
        coverage_rationale="2 merged sources",
        authority_summary=0.8,
        reliability_summary=0.7,
        diversity_summary=0.6,
        sources=sources,
        rejected_count=0,
        warnings=(),
        conflicts=(),
        stop_reason="sufficient_evidence",
        adapter_calls=("pubmed",),
    )


def _source(*, sid: str, title: str) -> EvidenceObject:
    return EvidenceObject(
        id=sid,
        source_id=sid,
        adapter="pubmed",
        retrieval_method="abstract",
        title=title,
        excerpt=f"Abstract for {title}",
        full_text=f"Abstract for {title}",
        url=f"https://example.org/{sid}",
        document_type="journal_abstract",
        relevance_score=0.7,
        authority_score=0.9,
        reliability_score=0.8,
        fetch_status="abstract",
    )


class TestDeepResearchSynthesis(unittest.TestCase):
    def test_strip_redundant_findings_heading(self) -> None:
        raw = "## Findings\n\nACE inhibitors reduce mortality. [1]"
        self.assertEqual(
            _strip_redundant_findings_heading(raw),
            "ACE inhibitors reduce mortality. [1]",
        )

    def test_numbered_retrieval_context(self) -> None:
        ui = [
            {"id": 1, "filename": "Trial A", "content": "Outcome improved."},
            {"id": 2, "filename": "Trial B", "content": "No benefit."},
        ]
        ctx = build_numbered_retrieval_context(ui)
        self.assertIn("--- [1]: Trial A ---", ctx)
        self.assertIn("--- [2]: Trial B ---", ctx)

    def test_synthesize_renumbers_citations(self) -> None:
        bundle = _bundle(
            sources=(
                _source(sid="ek_1", title="Trial A"),
                _source(sid="ek_2", title="Trial B"),
            )
        )

        def _gen(*, system: str, user: str) -> str:
            _ = system, user
            return "ACE inhibitors reduce mortality. [2] Another line. [1]"

        result = synthesize_deep_research_findings(
            "ACE inhibitors HF",
            bundle,
            generate_fn=_gen,
        )
        self.assertTrue(result.synthesized)
        self.assertIn("[1]", result.findings_markdown)
        self.assertEqual(len(result.ui_sources), 2)

    def test_compose_report_includes_findings_and_bibliography(self) -> None:
        bundle = _bundle(sources=(_source(sid="ek_1", title="Trial A"),))
        from core.knowledge.deep_research_synthesis import DeepResearchSynthesisResult

        synthesis = DeepResearchSynthesisResult(
            findings_markdown="Key finding. [1]",
            ui_sources=[{"id": 1, "filename": "Trial A", "content": "x"}],
            synthesized=True,
        )
        report = compose_deep_research_report(
            query="test query",
            bundle=bundle,
            sub_queries=("test query",),
            synthesis=synthesis,
        )
        self.assertIn("## Findings", report)
        self.assertIn("Key finding. [1]", report)
        self.assertIn("## Bibliography", report)
        self.assertIn("Trial A", report)


if __name__ == "__main__":
    unittest.main()
