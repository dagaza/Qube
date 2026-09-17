"""Tests for cross-source evidence conflict detection."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.knowledge.bundle_builder import build_general_web_bundle  # noqa: E402
from core.knowledge.conflicts.detect import (  # noqa: E402
    detect_epistemic_conflicts,
    detect_evidence_conflicts,
)
from core.knowledge.types import EvidenceObject  # noqa: E402
from core.memory_filters import EVIDENCE_CONFLICT_SYNTHESIS_SUFFIX  # noqa: E402
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt  # noqa: E402


def _web_source(
    index: int,
    *,
    title: str,
    excerpt: str,
) -> EvidenceObject:
    return EvidenceObject(
        id=f"ek_{index}",
        source_id=f"https://example.org/{index}",
        adapter="duckduckgo",
        retrieval_method="serp",
        title=title,
        excerpt=excerpt,
        full_text=None,
        url=f"https://example.org/{index}",
        document_type="web_snippet",
        relevance_score=0.5,
        authority_score=0.35,
        reliability_score=0.4,
        fetch_status="snippet_only",
    )


class EvidenceConflictTests(unittest.TestCase):
    def test_epistemic_conflict_detects_outcome_vs_schedule_mix(self) -> None:
        sources = (
            _web_source(
                1,
                title="2026 NBA Finals - Wikipedia",
                excerpt=(
                    "The series ended when the New York Knicks defeated "
                    "the San Antonio Spurs, four games to one."
                ),
            ),
            _web_source(
                2,
                title="2026 NBA Finals",
                excerpt="The 2026 NBA Finals are scheduled to tip off June 3 on ABC.",
            ),
        )
        conflicts = detect_epistemic_conflicts(sources, topic="2026 NBA Finals")
        self.assertEqual(len(conflicts), 1)
        labels = {label for label, _ in conflicts[0].positions}
        self.assertIn("retrospective_outcome", labels)
        self.assertIn("prospective", labels)

    def test_epistemic_conflict_absent_for_aligned_outcome_sources(self) -> None:
        sources = (
            _web_source(
                1,
                title="Final result",
                excerpt="The Knicks defeated the Spurs four games to one.",
            ),
            _web_source(
                2,
                title="Playoffs summary",
                excerpt="New York won the championship series in five games.",
            ),
        )
        self.assertEqual(detect_epistemic_conflicts(sources), ())

    def test_general_web_bundle_records_conflicts_and_warning(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="Who won the 2026 NBA Finals?",
            query_resolved="Who won the 2026 NBA Finals?",
            kept_rows=[
                {
                    "title": "2026 NBA Finals - Wikipedia",
                    "snippet": (
                        "The series ended when the New York Knicks defeated "
                        "the San Antonio Spurs, four games to one."
                    ),
                    "url": "https://example.org/wiki",
                    "_web_token_overlap": 0.4,
                },
                {
                    "title": "2026 NBA Finals",
                    "snippet": "The 2026 NBA Finals are scheduled to tip off June 3 on ABC.",
                    "url": "https://example.org/nba",
                    "_web_token_overlap": 0.35,
                },
            ],
            rejected_count=0,
            latency_ms=10.0,
        )
        self.assertEqual(len(bundle.conflicts), 1)
        self.assertIn("material_conflict", bundle.warnings)
        self.assertLessEqual(bundle.reliability_summary, 0.45)
        self.assertTrue(bundle.summary_for_skills().has_conflicts)

    def test_web_prompt_adds_conflict_synthesis_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_source_count=3,
            web_hit_count=3,
            evidence_has_conflicts=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(EVIDENCE_CONFLICT_SYNTHESIS_SUFFIX.strip()[:40], system)
        self.assertIn("Do NOT merge incompatible claims", system)


if __name__ == "__main__":
    unittest.main()
