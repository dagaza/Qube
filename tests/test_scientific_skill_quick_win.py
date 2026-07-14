"""Tests for scientific evidence skill auto-force on @evidence turns."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.skills import activate_skills  # noqa: E402
from core.skills.types import SkillContext, SkillSettings  # noqa: E402
from core.knowledge.types import (  # noqa: E402
    EvidenceBundle,
    EvidenceBundleSummary,
    EvidenceObject,
    SERVICE_SCIENTIFIC_EVIDENCE,
)


def _scientific_summary() -> EvidenceBundleSummary:
    return EvidenceBundleSummary(
        present=True,
        knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
        source_count=2,
        confidence=0.8,
        coverage="excellent",
        has_conflicts=False,
        warnings=(),
        source_types=("journal_abstract",),
        fetch_depth="abstract",
    )


class TestScientificSkillQuickWin(unittest.TestCase):
    def test_forced_scientific_research_when_skills_disabled(self) -> None:
        ctx = SkillContext(
            user_query="semaglutide trials",
            clean_query="semaglutide trials",
            execution_route="WEB",
            has_retrieval_sources=True,
            source_count=2,
            follow_up_active=False,
            explicit_remember_active=False,
            file_search_active=False,
            narrative_active=False,
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            evidence_summary=_scientific_summary(),
        )
        settings = SkillSettings(
            enabled=False,
            min_activation_score=0.55,
            max_active_skills=3,
            total_prompt_char_budget=1200,
            embedding_boost_enabled=True,
            debug_log_enabled=False,
        )
        result = activate_skills(
            ctx,
            settings=settings,
            forced_skill_ids=("scientific_research",),
        )
        self.assertIsNone(result.skipped_reason)
        self.assertEqual(len(result.activations), 1)
        self.assertEqual(result.activations[0].skill_id, "scientific_research")
        self.assertIn("Findings", result.prompt_block)


if __name__ == "__main__":
    unittest.main()
