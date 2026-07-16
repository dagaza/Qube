"""Tests for section chunking, ranking, and prompt budgeting (M4)."""

from __future__ import annotations

import os
import re
import sys
import time
import unittest
from pathlib import Path

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.document.types import (  # noqa: E402
    Document,
    DocumentMetadata,
    DocumentSection,
)
from core.knowledge.extractors.registry import extract_document  # noqa: E402
from core.knowledge.fetch.section_chunker import chunk_document  # noqa: E402
from core.knowledge.fetch.section_ranker import (  # noqa: E402
    document_to_evidence_objects,
    rank_section_chunks,
)
from core.knowledge.types import COVERAGE_ADEQUATE, EvidenceBundle  # noqa: E402
from core.knowledge.ui_adapter import bundle_to_prompt_context  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "fetch"


def _long_article_document() -> Document:
    sections = [
        DocumentSection(
            heading="Introduction",
            level=1,
            text=(
                "Birds across many habitats maintain feather health through dust bathing. "
                "This introduction explains why the behavior matters for parasite control."
            ),
        ),
        DocumentSection(
            heading="Why birds dust bathe",
            level=2,
            text=(
                "Dust absorbs excess oil and helps dislodge mites from plumage in dry habitats. "
                "The behavior is especially common when water bathing is unavailable."
            ),
        ),
        DocumentSection(
            heading="Common species",
            level=2,
            text="Sparrows, quail, and emus are well known for this behavior.",
        ),
        DocumentSection(
            heading="Garden tips",
            level=2,
            text=(
                "Provide a dry sandy patch in your garden so local birds can dust-bathe safely. "
                "Keep the patch away from feeders to reduce predator ambush risk."
            ),
        ),
        DocumentSection(
            heading="Ornithology notes",
            level=2,
            text=(
                "Researchers observe that dust bathing frequency rises during molting periods. "
                "Field studies link the habit to lower ectoparasite loads in arid climates."
            ),
        ),
        DocumentSection(
            heading="Unrelated finance summary",
            level=2,
            text=(
                "Quarterly bond yields moved higher after the central bank signaled tighter policy. "
                "Investors rotated into short-duration instruments during the trading session."
            ),
        ),
    ]
    return Document(
        url="https://example.com/birds/dust-bathing",
        title="Dust Bathing in Birds",
        sections=sections,
        metadata=DocumentMetadata(
            extractor_name="TrafilaturaExtractor",
            extractor_version="1.0.0",
            extractor_confidence=0.3,
            fetch_tier="http",
        ),
    )


def _bundle_from_sources(sources) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="bundle_test",
        query_raw="Why do birds take dust baths?",
        query_resolved="Why do birds take dust baths?",
        knowledge_service="general_web",
        retrieval_strategy="fetch_section_rank",
        profile_version="0.4.0",
        retrieved_at=time.time(),
        latency_ms=12.0,
        confidence=0.62,
        coverage=COVERAGE_ADEQUATE,
        coverage_rationale="Fetched sections ranked for query.",
        authority_summary=0.35,
        reliability_summary=0.55,
        diversity_summary=0.8,
        sources=tuple(sources),
        rejected_count=2,
        warnings=(),
        conflicts=(),
        stop_reason="budget_exhausted",
        adapter_calls=("duckduckgo", "fetch_engine"),
    )


class TestSectionRanker(unittest.TestCase):
    def test_chunk_document_splits_oversized_sections(self) -> None:
        long_text = " ".join(["Paragraph about feather maintenance."] * 80)
        document = Document(
            url="https://example.com/long",
            title="Long article",
            sections=[DocumentSection(heading="Details", level=2, text=long_text)],
        )
        chunks = chunk_document(document, max_section_chars=400)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in chunks))

    def test_rank_section_chunks_returns_top_three(self) -> None:
        document = _long_article_document()
        chunks = chunk_document(document, max_section_chars=800)
        ranked = rank_section_chunks(
            chunks,
            document=document,
            query="Why do birds take dust baths?",
            max_results=3,
        )
        self.assertEqual(len(ranked), 3)
        self.assertGreater(ranked[0].relevance_score, 0.0)

    def test_document_to_evidence_objects_maps_sections(self) -> None:
        document = _long_article_document()
        evidence = document_to_evidence_objects(
            document,
            query="Why do birds take dust baths?",
            max_results=3,
        )
        self.assertEqual(len(evidence), 3)
        self.assertTrue(all(obj.fetch_status == "full_extract" for obj in evidence))
        self.assertTrue(all(obj.document_type == "web_section" for obj in evidence))
        self.assertTrue(all(obj.excerpt for obj in evidence))
        self.assertIn("extractor_name", evidence[0].raw_metadata)

    def test_prompt_context_respects_budget_without_mid_word_chop(self) -> None:
        document = _long_article_document()
        evidence = document_to_evidence_objects(
            document,
            query="Why do birds take dust baths?",
            max_results=3,
        )
        bundle = _bundle_from_sources(evidence)
        char_budget = 420
        prompt = bundle_to_prompt_context(bundle, char_budget=char_budget)
        full_prompt = bundle_to_prompt_context(bundle, char_budget=0)

        self.assertIn("WEB SEARCH RESULTS", prompt)
        self.assertLessEqual(len(prompt), char_budget)

        if len(prompt) < len(full_prompt):
            trailing_word = re.findall(r"\w+$", prompt)
            if trailing_word:
                self.assertIn(trailing_word[0], full_prompt)

    def test_fixture_article_pipeline_end_to_end(self) -> None:
        try:
            import trafilatura  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("trafilatura is not installed") from exc

        html = (_FIXTURES / "article_clean.html").read_text(encoding="utf-8")
        document = extract_document(html, "https://example.com/birds")
        evidence = document_to_evidence_objects(
            document,
            query="Why do birds take dust baths?",
            max_results=3,
        )
        self.assertGreaterEqual(len(evidence), 1)
        bundle = _bundle_from_sources(evidence)
        prompt = bundle_to_prompt_context(bundle, char_budget=600)
        self.assertLessEqual(len(prompt), 600)
        self.assertRegex(prompt.lower(), r"bird|dust|feather|parasite")


if __name__ == "__main__":
    unittest.main()
