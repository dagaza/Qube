"""Tests for Phase 5 slice 3 transparency and citation export."""

from __future__ import annotations

import json
import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.evidence_citations import (  # noqa: E402
    source_to_apa,
    source_to_bibtex,
    sources_to_bibtex,
)
from core.knowledge.evidence_transparency import build_evidence_transparency  # noqa: E402
from core.knowledge.types import EvidenceBundle, EvidenceObject  # noqa: E402
from core.knowledge.ui_adapter import evidence_to_ui_source  # noqa: E402
from core.knowledge.ui_sources_payload import (  # noqa: E402
    decode_sources_payload,
    encode_sources_payload,
)


def _bundle(*, sources: tuple[EvidenceObject, ...]) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1",
        query_raw="ACE inhibitors heart failure",
        query_resolved="ACE inhibitors heart failure",
        knowledge_service="scientific_evidence",
        retrieval_strategy="deep_research_merged",
        profile_version="0.1.0",
        retrieved_at=0.0,
        latency_ms=12.0,
        confidence=0.84,
        coverage="excellent",
        coverage_rationale="5 sources across 2 indexes",
        authority_summary=0.8,
        reliability_summary=0.7,
        diversity_summary=0.6,
        sources=sources,
        rejected_count=2,
        warnings=("preprint_included",),
        conflicts=(),
        stop_reason="sufficient_evidence",
        adapter_calls=("pubmed", "openalex"),
    )


class TestEvidenceTransparencySlice3(unittest.TestCase):
    def test_ui_source_includes_scores_and_fetch_status(self) -> None:
        obj = EvidenceObject(
            id="ek_1",
            source_id="ek_1",
            adapter="pubmed",
            retrieval_method="abstract",
            title="ACE inhibitors trial",
            excerpt="Abstract text",
            full_text="Abstract text",
            url="https://example.org/1",
            document_type="journal_abstract",
            doi="10.1/test",
            venue="Circulation",
            authors=("Smith, J.",),
            publication_date="2024",
            relevance_score=0.81,
            authority_score=0.9,
            reliability_score=0.75,
            fetch_status="abstract",
            preprint=False,
        )
        row = evidence_to_ui_source(obj, ui_id=1)
        self.assertEqual(row["fetch_status"], "abstract")
        self.assertEqual(row["relevance_score"], 0.81)
        self.assertEqual(row["venue"], "Circulation")

    def test_build_evidence_transparency_summary(self) -> None:
        bundle = _bundle(
            sources=(
                EvidenceObject(
                    id="ek_1",
                    source_id="ek_1",
                    adapter="pubmed",
                    retrieval_method="abstract",
                    title="ACE trial",
                    excerpt="x",
                    full_text="x",
                    url=None,
                    document_type="journal_abstract",
                    fetch_status="abstract",
                ),
            )
        )
        summary = build_evidence_transparency(
            bundle,
            diagnostics={
                "merged_sources_pre_filter": 8,
                "merged_sources_post_filter": 5,
                "merged_relevance_dropped": 3,
            },
            sub_queries=("ACE inhibitors", "ACE inhibitors RCT"),
        )
        self.assertIn("why_summary", summary)
        self.assertIn("Merge filter", summary["why_summary"])
        self.assertEqual(summary["source_count"], 1)

    def test_bibtex_and_apa_export(self) -> None:
        src = {
            "id": 1,
            "evidence_id": "ek_1",
            "filename": "ACE inhibitors in heart failure",
            "authors": ["Smith, J.", "Doe, A."],
            "venue": "Circulation",
            "publication_date": "2024-06",
            "doi": "10.1/example",
        }
        self.assertIn("@article{", source_to_bibtex(src))
        self.assertIn("ACE inhibitors in heart failure", source_to_apa(src))
        self.assertIn("Smith, J.", sources_to_bibtex([src]))

    def test_sources_payload_v2_roundtrip(self) -> None:
        sources = [{"id": 1, "filename": "Trial A", "type": "web"}]
        transparency = {"why_summary": "Coverage: excellent"}
        raw = encode_sources_payload(sources, transparency=transparency)
        assert raw is not None
        decoded_sources, decoded_transparency = decode_sources_payload(raw)
        self.assertEqual(len(decoded_sources), 1)
        assert decoded_transparency is not None
        self.assertEqual(decoded_transparency["why_summary"], "Coverage: excellent")
        legacy = decode_sources_payload(json.dumps(sources))
        self.assertEqual(len(legacy[0]), 1)
        self.assertIsNone(legacy[1])


if __name__ == "__main__":
    unittest.main()
