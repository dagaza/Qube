"""Tests for Phase 6 Slice 3 entity resolution (ADR 002 compositional registry)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research import _dedupe_sources, merge_evidence_bundles  # noqa: E402
from core.knowledge.entities.activation import resolve_active_components  # noqa: E402
from core.knowledge.entities.enrich import enrich_evidence_object  # noqa: E402
from core.knowledge.entities.pipeline import resolve_entities_from_text  # noqa: E402
from core.knowledge.entities.registry import ALWAYS_ON_EXTRACTOR_IDS  # noqa: E402
from core.knowledge.entities.types import EntityResolutionContext  # noqa: E402
from core.knowledge.evidence_transparency import build_evidence_transparency  # noqa: E402
from core.knowledge.types import EvidenceBundle, EvidenceObject  # noqa: E402


def _source(
    *,
    sid: str,
    title: str,
    excerpt: str = "",
    doi: str | None = None,
    url: str | None = None,
    relevance_score: float = 0.7,
) -> EvidenceObject:
    return EvidenceObject(
        id=sid,
        source_id=sid,
        adapter="pubmed",
        retrieval_method="abstract",
        title=title,
        excerpt=excerpt or title,
        full_text=excerpt or title,
        url=url or f"https://example.org/{sid}",
        document_type="journal_abstract",
        doi=doi,
        relevance_score=relevance_score,
        authority_score=0.9,
        reliability_score=0.8,
        fetch_status="abstract",
    )


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


class TestEntityResolution(unittest.TestCase):
    def test_resolve_ace_and_hf_entities(self) -> None:
        text = "ACE inhibitors reduce mortality in heart failure (EMPEROR-Reduced trial)"
        ids = resolve_entities_from_text(text)
        joined = " ".join(ids)
        self.assertIn("entity:drug-class:ace-inhibitors", joined)
        self.assertIn("entity:condition:heart-failure", joined)
        self.assertIn("entity:trial:emperor-reduced", joined)

    def test_enrich_attaches_entity_ids(self) -> None:
        src = _source(
            sid="ek_1",
            title="SGLT2 inhibitors in heart failure",
            excerpt="Dapagliflozin outcomes in HFrEF.",
        )
        enriched = enrich_evidence_object(src)
        self.assertTrue(enriched.entity_ids)
        joined = " ".join(enriched.entity_ids)
        self.assertIn("entity:drug-class:sglt2-inhibitors", joined)
        self.assertIn("entity:drug:dapagliflozin", joined)

    def test_merge_dedupes_by_trial_cluster_without_doi(self) -> None:
        a = _source(
            sid="ek_1",
            title="EMPEROR-Reduced: empagliflozin in heart failure",
            url="https://pubmed.ncbi.nlm.nih.gov/33000001/",
        )
        b = _source(
            sid="ek_2",
            title="Empagliflozin EMPEROR-Reduced outcomes summary",
            url="https://pubmed.ncbi.nlm.nih.gov/33000001/",
        )
        other = _source(
            sid="ek_3",
            title="SGLT2 inhibitors meta-analysis in heart failure",
        )
        merged = merge_evidence_bundles(
            query="SGLT2 inhibitors heart failure",
            bundles=(
                _bundle(sources=(a,)),
                _bundle(sources=(b, other)),
            ),
        )
        assert merged is not None
        self.assertEqual(len(merged.sources), 2)
        titles = {s.title for s in merged.sources}
        self.assertIn("SGLT2 inhibitors meta-analysis in heart failure", titles)

    def test_distinct_drug_classes_not_deduped(self) -> None:
        ace = _source(
            sid="ek_1",
            title="ACE inhibitors reduce mortality in heart failure",
        )
        sglt2 = _source(
            sid="ek_2",
            title="SGLT2 inhibitors reduce hospitalization in heart failure",
        )
        merged = merge_evidence_bundles(
            query="heart failure therapies",
            bundles=(_bundle(sources=(ace, sglt2)),),
        )
        assert merged is not None
        self.assertEqual(len(merged.sources), 2)

    def test_dedupe_by_doi_still_wins(self) -> None:
        low = _source(
            sid="ek_1",
            title="ACE inhibitors trial",
            doi="10.1/ace.hf",
            relevance_score=0.5,
        )
        high = _source(
            sid="ek_2",
            title="ACE inhibitors trial copy",
            doi="10.1/ace.hf",
            relevance_score=0.95,
        )
        deduped = _dedupe_sources([low, high])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].relevance_score, 0.95)

    def test_transparency_lists_entities(self) -> None:
        src = enrich_evidence_object(
            _source(
                sid="ek_1",
                title="ACE inhibitors in heart failure",
            )
        )
        bundle = _bundle(
            sources=(src,),
            query="ACE inhibitors heart failure",
        )
        summary = build_evidence_transparency(bundle)
        self.assertIn("Entities detected", summary["why_summary"])
        self.assertIn("entity_ids", summary)


class TestEntityActivation(unittest.TestCase):
    def test_bibliographic_extractor_always_on(self) -> None:
        ctx = EntityResolutionContext(
            query_resolved="unrelated general query",
            knowledge_service="general_web",
        )
        active = resolve_active_components(ctx)
        for extractor_id in ALWAYS_ON_EXTRACTOR_IDS:
            self.assertIn(extractor_id, active.extractor_ids)

    def test_scientific_non_medical_query_skips_biomedical_extractors(self) -> None:
        ctx = EntityResolutionContext(
            query_resolved="transformer attention mechanism neural machine translation",
            knowledge_service="scientific_evidence",
        )
        active = resolve_active_components(ctx)
        self.assertNotIn("biomedical_drugs", active.extractor_ids)

    def test_scientific_medical_query_activates_biomedical(self) -> None:
        ctx = EntityResolutionContext(
            query_resolved="ACE inhibitors reduce mortality in heart failure",
            knowledge_service="scientific_evidence",
        )
        active = resolve_active_components(ctx)
        self.assertIn("biomedical_drugs", active.extractor_ids)

    def test_query_activation_without_service_hint(self) -> None:
        ctx = EntityResolutionContext(
            query_resolved="machine learning benchmarks",
            knowledge_service="general_web",
        )
        active = resolve_active_components(ctx)
        self.assertNotIn("biomedical_drugs", active.extractor_ids)

    def test_pubmed_source_activates_biomedical(self) -> None:
        ctx = EntityResolutionContext(
            query_resolved="",
            knowledge_service="general_web",
        )
        src = _source(sid="ek_1", title="Neutral title")
        active = resolve_active_components(ctx, source=src)
        self.assertIn("biomedical_drugs", active.extractor_ids)


if __name__ == "__main__":
    unittest.main()
