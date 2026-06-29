"""Tests for Phase 4 deep research scaffold."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research import (  # noqa: E402
    DeepResearchCancelled,
    _dedupe_sources,
    apply_merged_relevance_gate,
    build_bibliography_report,
    decompose_query,
    merge_evidence_bundles,
    run_deep_research,
)
from core.knowledge.deep_research_decompose import normalize_deep_research_query
from core.knowledge.types import EvidenceObject, EvidenceBundle  # noqa: E402


def _bundle(*, sources: tuple[EvidenceObject, ...], query: str = "test") -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1",
        query_raw=query,
        query_resolved=query,
        knowledge_service="scientific_evidence",
        retrieval_strategy="pubmed_openalex_arxiv_ranked",
        profile_version="0.4.0",
        retrieved_at=0.0,
        latency_ms=10.0,
        confidence=0.8,
        coverage="adequate",
        coverage_rationale="test",
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


def _source(
    *,
    sid: str,
    title: str,
    doi: str | None = None,
    relevance_score: float = 0.7,
) -> EvidenceObject:
    return EvidenceObject(
        id=sid,
        source_id=sid,
        adapter="pubmed",
        retrieval_method="abstract",
        title=title,
        excerpt=title,
        full_text=title,
        url=f"https://example.org/{sid}",
        document_type="journal_abstract",
        doi=doi,
        relevance_score=relevance_score,
        authority_score=0.9,
        reliability_score=0.8,
        fetch_status="abstract",
    )


class TestDeepResearch(unittest.TestCase):
    def test_decompose_expands_clinical_query(self) -> None:
        parts = decompose_query("semaglutide cardiovascular outcomes")
        self.assertGreaterEqual(len(parts), 2)
        self.assertIn("semaglutide cardiovascular outcomes", parts[0])

    def test_decompose_normalizes_mace_typo(self) -> None:
        parts = decompose_query("MACE inhibitors heart failure evidence")
        self.assertTrue(all("ACE inhibitors" in p for p in parts))
        self.assertNotIn("MACE", " ".join(parts))

    def test_decompose_splits_on_and(self) -> None:
        parts = decompose_query(
            "semaglutide cardiovascular outcomes and heart failure hospitalization"
        )
        self.assertGreaterEqual(len(parts), 2)

    def test_merge_dedupes_by_doi(self) -> None:
        src = _source(sid="ek_1", title="Trial A", doi="10.1/test")
        dup = _source(sid="ek_2", title="Trial A copy", doi="10.1/test")
        other = _source(sid="ek_3", title="Trial B", doi="10.1/other")
        merged = merge_evidence_bundles(
            query="test",
            bundles=(
                _bundle(sources=(src,)),
                _bundle(sources=(dup, other)),
            ),
        )
        assert merged is not None
        self.assertEqual(len(merged.sources), 2)

    def test_report_includes_bibliography(self) -> None:
        bundle = _bundle(sources=(_source(sid="ek_1", title="Semaglutide trial"),))
        report = build_bibliography_report(
            query="semaglutide",
            bundle=bundle,
            sub_queries=("semaglutide",),
        )
        self.assertIn("# Deep Research Report", report)
        self.assertIn("Semaglutide trial", report)

    def test_merged_relevance_gate_drops_tangential(self) -> None:
        query = "ACE inhibitors heart failure mortality"
        bundle = _bundle(
            sources=(
                _source(
                    sid="ek_1",
                    title="ACE inhibitors reduce mortality in heart failure",
                ),
                _source(
                    sid="ek_2",
                    title="Takotsubo cardiomyopathy stress cardiomyopathy review",
                ),
            ),
            query=query,
        )
        filtered, dropped, diag = apply_merged_relevance_gate(query=query, bundle=bundle)
        assert filtered is not None
        titles = [s.title for s in filtered.sources]
        self.assertIn("ACE inhibitors reduce mortality in heart failure", titles)
        self.assertNotIn("Takotsubo cardiomyopathy stress cardiomyopathy review", titles)
        self.assertGreaterEqual(dropped, 1)
        self.assertIn("merged_anchor_tokens", diag)

    def test_merged_relevance_gate_reorders_when_nothing_dropped(self) -> None:
        query = "statin primary prevention cardiovascular risk meta-analysis"
        bundle = _bundle(
            sources=(
                _source(
                    sid="ek_1",
                    title="Statins for the primary prevention of cardiovascular disease",
                    relevance_score=1.0,
                ),
                _source(
                    sid="ek_2",
                    title="Cardiovascular Outcomes in Individuals With Diabetes",
                    relevance_score=0.57,
                ),
                _source(
                    sid="ek_3",
                    title="Evaluating the effectiveness of simvastatin in multiple sclerosis",
                    relevance_score=0.55,
                ),
            ),
            query=query,
        )
        filtered, dropped, diag = apply_merged_relevance_gate(query=query, bundle=bundle)
        assert filtered is not None
        self.assertEqual(dropped, 0)
        self.assertEqual(diag.get("merged_ranker_version"), "2.0")
        titles = [s.title for s in filtered.sources]
        self.assertEqual(titles[0], "Statins for the primary prevention of cardiovascular disease")
        self.assertIn("simvastatin", titles[1].lower())

    def test_dedupe_keeps_highest_scored_duplicate_title(self) -> None:
        low = _source(
            sid="ek_1",
            title="ACE inhibitors in heart failure",
            doi="10.1/ace",
            relevance_score=0.5,
        )
        high = _source(
            sid="ek_2",
            title="ACE inhibitors in heart failure",
            doi="10.1/ace",
            relevance_score=0.9,
        )
        deduped = _dedupe_sources([low, high])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].relevance_score, 0.9)

    @patch("core.knowledge.deep_research.run_v2_web_retrieval")
    def test_run_deep_research_respects_cancel(self, mock_retrieve) -> None:
        from core.knowledge.types import WebRetrievalOutcome

        bundle = _bundle(
            sources=(_source(sid="ek_1", title="Heart outcomes"),),
            query="heart",
        )
        mock_retrieve.return_value = WebRetrievalOutcome(
            web_results=None,
            web_results_raw_for_audit=None,
            web_results_kept_for_audit=None,
            relevance_diag=None,
            skip_enrichment=False,
            bundle=bundle,
            latency_ms=5.0,
        )
        cancelled = {"value": False}

        def should_cancel() -> bool:
            cancelled["value"] = True
            return True

        with self.assertRaises(DeepResearchCancelled):
            run_deep_research(
                "heart outcomes and kidney outcomes",
                should_cancel=should_cancel,
            )

    @patch("core.knowledge.deep_research.run_v2_web_retrieval")
    def test_run_deep_research_merges_subqueries(self, mock_retrieve) -> None:
        from core.knowledge.types import WebRetrievalOutcome

        bundle = _bundle(
            sources=(
                _source(sid="ek_1", title="Kidney outcomes in heart failure cohort"),
            ),
            query="heart",
        )

        mock_retrieve.return_value = WebRetrievalOutcome(
            web_results=None,
            web_results_raw_for_audit=None,
            web_results_kept_for_audit=None,
            relevance_diag=None,
            skip_enrichment=False,
            bundle=bundle,
            latency_ms=5.0,
        )
        result = run_deep_research(
            "heart outcomes and kidney outcomes",
        )
        self.assertGreaterEqual(mock_retrieve.call_count, 1)
        self.assertIsNotNone(result.merged_bundle)
        self.assertIn("Bibliography", result.report_markdown)


if __name__ == "__main__":
    unittest.main()
