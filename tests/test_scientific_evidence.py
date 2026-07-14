"""Tests for Phase 2 scientific evidence pipeline."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.composer_attachments import (  # noqa: E402
    ComposerAttachment,
    resolve_attachment_routing,
)
from core.knowledge.registry import resolve_turn_knowledge_service  # noqa: E402
from core.knowledge.conflicts.detect import detect_conflicts  # noqa: E402
from core.knowledge.types import (  # noqa: E402
    EvidenceObject,
    SERVICE_SCIENTIFIC_EVIDENCE,
)
from core.knowledge.web_retrieval import run_v2_web_retrieval  # noqa: E402
from core.skills.registry import get_skill  # noqa: E402

_PUBMED_ROW = {
    "title": "Semaglutide and cardiovascular outcomes",
    "snippet": "Semaglutide was effective at reducing major adverse cardiovascular events.",
    "full_text": "Semaglutide was effective at reducing major adverse cardiovascular events in adults with type 2 diabetes.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    "_adapter": "pubmed",
    "authors": ("Smith, J",),
    "venue": "NEJM",
    "publication_date": "2024-01",
    "doi": "10.1000/test.1",
    "peer_reviewed": True,
    "preprint": False,
    "document_type": "journal_abstract",
}

_OPENALEX_ROW = {
    "title": "No benefit from semaglutide in subgroup",
    "snippet": "No significant benefit was observed for the primary endpoint in this cohort.",
    "full_text": "No significant benefit was observed for the primary endpoint in this cohort analysis.",
    "url": "https://openalex.org/W123",
    "_adapter": "openalex",
    "authors": ("Lee, A",),
    "venue": "Lancet",
    "publication_date": "2023",
    "doi": "10.1000/test.2",
    "peer_reviewed": True,
    "preprint": False,
    "document_type": "journal_abstract",
}


class TestScientificEvidence(unittest.TestCase):
    def test_composer_evidence_routing(self) -> None:
        patch = resolve_attachment_routing(
            [ComposerAttachment(kind="tool", id="evidence", label="Scientific literature")]
        )
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["attachment_tool"], "evidence")

    def test_composer_science_alias_routing(self) -> None:
        patch = resolve_attachment_routing(
            [ComposerAttachment(kind="tool", id="science", label="Scientific literature")]
        )
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["attachment_tool"], "science")

    def test_resolve_science_alias_service(self) -> None:
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="science"),
            SERVICE_SCIENTIFIC_EVIDENCE,
        )
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="evidence"),
            SERVICE_SCIENTIFIC_EVIDENCE,
        )

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    @patch.dict(
        "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
        {
            "pubmed": lambda q, max_results=3: [_PUBMED_ROW],
            "openalex": lambda q, max_results=3: [_OPENALEX_ROW],
            "arxiv": lambda q, max_results=3: [],
        },
        clear=False,
    )
    def test_v2_scientific_bundle(self, _set_cache, _get_cache) -> None:
        outcome = run_v2_web_retrieval(
            query="semaglutide cardiovascular outcomes",
            semantic_query="semaglutide cardiovascular outcomes",
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
        )

        self.assertFalse(outcome.skip_enrichment)
        assert outcome.bundle is not None
        self.assertEqual(outcome.bundle.knowledge_service, SERVICE_SCIENTIFIC_EVIDENCE)
        self.assertIn("pubmed", outcome.bundle.adapter_calls)
        self.assertGreaterEqual(len(outcome.bundle.sources), 1)
        self.assertTrue(outcome.bundle.sources[0].doi)

    def test_conflict_detection(self) -> None:
        sources = (
            EvidenceObject(
                id="ek_1",
                source_id="a",
                adapter="pubmed",
                retrieval_method="abstract",
                title="Trial A",
                excerpt="Semaglutide was effective and safe.",
                full_text="Semaglutide was effective and safe.",
                url="https://example.org/a",
                document_type="journal_abstract",
            ),
            EvidenceObject(
                id="ek_2",
                source_id="b",
                adapter="openalex",
                retrieval_method="abstract",
                title="Trial B",
                excerpt="No significant benefit was observed.",
                full_text="No significant benefit was observed.",
                url="https://example.org/b",
                document_type="journal_abstract",
            ),
        )
        conflicts = detect_conflicts(sources, topic="semaglutide")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].severity, "material")

    def test_scientific_research_skill_registered(self) -> None:
        skill = get_skill("scientific_research")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertEqual(skill.id, "scientific_research")


if __name__ == "__main__":
    unittest.main()
