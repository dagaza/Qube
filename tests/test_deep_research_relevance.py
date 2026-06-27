"""Tests for Phase 5 deep-research relevance scoring."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research_merge import (  # noqa: E402
    extract_query_anchor_tokens,
    filter_merged_sources_for_query,
    source_passes_anchor_gate,
)
from core.knowledge.deep_research_relevance import (  # noqa: E402
    build_merge_relevance_diag,
    score_merged_bundle_relevance,
    source_title_is_relevant,
)
from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace
from core.knowledge.types import EvidenceBundle, EvidenceObject  # noqa: E402


def _source(title: str, *, sid: str = "ek_1") -> EvidenceObject:
    return EvidenceObject(
        id=sid,
        source_id=sid,
        adapter="pubmed",
        retrieval_method="abstract",
        title=title,
        excerpt=title,
        full_text=title,
        url="https://example.org",
        document_type="journal_abstract",
        relevance_score=0.7,
        authority_score=0.8,
        reliability_score=0.7,
        fetch_status="abstract",
    )


def _bundle(*, sources: tuple[EvidenceObject, ...]) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1",
        query_raw="test",
        query_resolved="test",
        knowledge_service="scientific_evidence",
        retrieval_strategy="deep_research_merged",
        profile_version="0.1.0",
        retrieved_at=0.0,
        latency_ms=1.0,
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


class TestDeepResearchRelevance(unittest.TestCase):
    def test_extract_ace_anchor_tokens(self) -> None:
        anchors = extract_query_anchor_tokens(
            "ACE inhibitors heart failure mortality evidence"
        )
        self.assertIn("ace", anchors)
        self.assertIn("angiotensin", anchors)

    def test_anchor_gate_drops_chemo_cardiotoxicity(self) -> None:
        anchors = extract_query_anchor_tokens("ACE inhibitors heart failure evidence")
        self.assertFalse(
            source_passes_anchor_gate(
                "Cardioprotective role of antihypertensive treatment in "
                "chemotherapy-induced cardiotoxicity umbrella review",
                anchors,
            )
        )
        self.assertTrue(
            source_passes_anchor_gate(
                "ACE inhibitors reduce mortality in chronic heart failure",
                anchors,
            )
        )

    def test_filter_drops_chemo_even_when_excerpt_mentions_ace(self) -> None:
        from core.knowledge.types import EvidenceObject

        chemo = EvidenceObject(
            id="ek_chemo",
            source_id="ek_chemo",
            adapter="pubmed",
            retrieval_method="abstract",
            title=(
                "Cardioprotective role of antihypertensive treatment in "
                "chemotherapy-induced cardiotoxicity umbrella review"
            ),
            excerpt="ACE inhibitors (ACEIs) are proposed as prophylactic therapies.",
            full_text="",
            url="",
            document_type="journal_abstract",
            relevance_score=1.0,
            authority_score=0.92,
            reliability_score=0.7,
            fetch_status="abstract",
        )
        on_topic = EvidenceObject(
            id="ek_hf",
            source_id="ek_hf",
            adapter="pubmed",
            retrieval_method="abstract",
            title="ACE inhibitors reduce mortality in chronic heart failure",
            excerpt="ACE inhibitors reduce mortality in chronic heart failure",
            full_text="",
            url="",
            document_type="journal_abstract",
            relevance_score=0.9,
            authority_score=0.92,
            reliability_score=0.7,
            fetch_status="abstract",
        )
        kept, dropped, diag = filter_merged_sources_for_query(
            "ACE inhibitors heart failure evidence",
            [chemo, on_topic],
        )
        titles = [s.title for s in kept]
        self.assertNotIn(chemo.title, titles)
        self.assertIn(on_topic.title, titles)
        self.assertGreaterEqual(diag.get("merged_title_reject_dropped", 0), 1)

    def test_filter_drops_tangential_without_ace_anchor(self) -> None:
        from core.knowledge.types import EvidenceObject

        sources = [
            EvidenceObject(
                id="ek_1",
                source_id="ek_1",
                adapter="pubmed",
                retrieval_method="abstract",
                title="ACE inhibitors reduce mortality in heart failure",
                excerpt="ACE inhibitors reduce mortality in heart failure",
                full_text="",
                url="",
                document_type="journal_abstract",
                relevance_score=0.7,
                authority_score=0.8,
                reliability_score=0.7,
                fetch_status="abstract",
            ),
            EvidenceObject(
                id="ek_2",
                source_id="ek_2",
                adapter="pubmed",
                retrieval_method="abstract",
                title="Takotsubo Syndrome review",
                excerpt="Takotsubo Syndrome review",
                full_text="",
                url="",
                document_type="journal_abstract",
                relevance_score=0.7,
                authority_score=0.8,
                reliability_score=0.7,
                fetch_status="abstract",
            ),
            EvidenceObject(
                id="ek_3",
                source_id="ek_3",
                adapter="pubmed",
                retrieval_method="abstract",
                title="Heart failure hospitalization trends",
                excerpt="Heart failure hospitalization trends",
                full_text="",
                url="",
                document_type="journal_abstract",
                relevance_score=0.7,
                authority_score=0.8,
                reliability_score=0.7,
                fetch_status="abstract",
            ),
        ]
        kept, dropped, diag = filter_merged_sources_for_query(
            "ACE inhibitors heart failure evidence",
            sources,
        )
        titles = [s.title for s in kept]
        self.assertIn("ACE inhibitors reduce mortality in heart failure", titles)
        self.assertNotIn("Takotsubo Syndrome review", titles)
        self.assertNotIn("Heart failure hospitalization trends", titles)
        self.assertGreaterEqual(dropped, 2)
        self.assertGreaterEqual(diag.get("merged_anchor_dropped", 0), 2)

    def test_reject_pattern_excludes_takotsubo(self) -> None:
        self.assertFalse(
            source_title_is_relevant(
                "Takotsubo Syndrome review",
                expect_any_tokens=["ace", "angiotensin"],
                reject_title_patterns=["takotsubo"],
            )
        )

    def test_expect_token_required_when_configured(self) -> None:
        self.assertTrue(
            source_title_is_relevant(
                "ACE inhibitors in heart failure",
                expect_any_tokens=["ace", "angiotensin"],
                reject_title_patterns=["takotsubo"],
            )
        )
        self.assertFalse(
            source_title_is_relevant(
                "Heart failure hospitalization trends",
                expect_any_tokens=["ace", "angiotensin"],
                reject_title_patterns=["takotsubo"],
            )
        )

    def test_score_merged_bundle_relevance(self) -> None:
        sources = (
            _source("ACE inhibitors reduce mortality in heart failure", sid="ek_1"),
            _source("Angiotensin receptor blockers in HF", sid="ek_2"),
            _source("Takotsubo cardiomyopathy review", sid="ek_3"),
        )
        result = score_merged_bundle_relevance(
            sources,
            expect_any_tokens=["ace", "angiotensin"],
            reject_title_patterns=["takotsubo"],
            top_n=3,
            min_relevant_in_top=2,
        )
        self.assertTrue(result["relevance_ok"])
        self.assertEqual(result["relevant_in_top"], 2)

    def test_build_merge_relevance_diag(self) -> None:
        diag = build_merge_relevance_diag(
            {
                "merged_relevance_dropped": 3,
                "merged_sources_pre_filter": 10,
                "merged_sources_post_filter": 7,
                "merged_anchor_tokens": ["ace", "angiotensin"],
                "merged_anchor_dropped": 5,
                "merged_semantic_gate": False,
            }
        )
        self.assertEqual(diag["merged_relevance_dropped"], 3)
        self.assertEqual(diag["merged_anchor_dropped"], 5)
        self.assertIn("merged_relevance_min_overlap", diag)

    def test_retrieval_trace_serializes_merge_relevance_diag(self) -> None:
        bundle = _bundle(sources=(_source("ACE inhibitors trial"),))
        trace = build_retrieval_trace(
            bundle,
            relevance_diag={
                "merged_relevance_dropped": 2,
                "merged_sources_pre_filter": 8,
                "merged_sources_post_filter": 6,
                "merged_relevance_min_overlap": 0.2,
            },
        )
        payload = serialize_retrieval_trace(trace, sources=bundle.sources)
        self.assertEqual(payload["relevance_diag"]["merged_relevance_dropped"], 2)
        self.assertEqual(payload["relevance_diag"]["merged_sources_post_filter"], 6)


if __name__ == "__main__":
    unittest.main()
