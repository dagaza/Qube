"""Tests for Merge Ranker v2 (weighted deep-research merge scoring)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research_merge import (  # noqa: E402
    rank_merged_sources_for_query,
)
from core.knowledge.ranking.merged_source import score_merged_source  # noqa: E402
from core.knowledge.types import EvidenceObject  # noqa: E402


def _src(
    title: str,
    *,
    sid: str = "ek_1",
    excerpt: str | None = None,
    relevance: float = 0.7,
) -> EvidenceObject:
    body = excerpt if excerpt is not None else title
    return EvidenceObject(
        id=sid,
        source_id=sid,
        adapter="pubmed",
        retrieval_method="abstract",
        title=title,
        excerpt=body,
        full_text=body,
        url="https://example.org",
        document_type="journal_abstract",
        relevance_score=relevance,
        authority_score=0.92,
        reliability_score=0.7,
        fetch_status="abstract",
    )


class TestMergeRankerV2(unittest.TestCase):
    def test_qa_session_titles_keep_multiple_sources(self) -> None:
        """Regression for QA-6B-G: HF papers without ACE in title can rank via excerpt/overlap."""
        query = "ACE inhibitors heart failure mortality evidence"
        sources = [
            _src(
                "Finerenone and Cardiorenal Outcomes: A Narrative Review.",
                sid="ek_1",
            ),
            _src(
                "Gaps in Guideline-Directed Medical Therapy for Heart Failure: A Call to Action.",
                sid="ek_2",
                excerpt="GDMT includes ACE inhibitors, beta blockers, and MRAs for HFrEF.",
            ),
            _src(
                "The Effect of Digoxin on Mortality and Morbidity in Patients with Heart Failure.",
                sid="ek_3",
                excerpt="Digoxin in heart failure; comparison with ACE inhibitor therapy.",
            ),
            _src(
                "Overview of randomized trials of angiotensin-converting enzyme inhibitors in heart failure.",
                sid="ek_4",
                relevance=1.0,
            ),
        ]
        kept, dropped, diag = rank_merged_sources_for_query(query, sources)
        self.assertGreaterEqual(len(kept), 2)
        self.assertEqual(diag.get("merged_ranker_version"), "2.0")
        self.assertFalse(diag.get("merged_title_first_gate"))
        titles = [s.title for s in kept]
        self.assertTrue(
            any("angiotensin-converting enzyme" in t for t in titles)
        )

    def test_reject_pattern_still_hard_drops_takotsubo(self) -> None:
        query = "ACE inhibitors heart failure evidence"
        sources = [
            _src("ACE inhibitors reduce mortality in heart failure", sid="ek_1"),
            _src("Takotsubo Syndrome review", sid="ek_2"),
        ]
        kept, _dropped, diag = rank_merged_sources_for_query(query, sources)
        titles = [s.title for s in kept]
        self.assertNotIn("Takotsubo Syndrome review", titles)
        self.assertGreaterEqual(diag.get("merged_title_reject_dropped", 0), 1)

    def test_excerpt_anchor_scores_higher_than_title_only_hf(self) -> None:
        query = "ACE inhibitors heart failure mortality evidence"
        anchors = ("ace", "angiotensin")
        gdmt = _src(
            "Gaps in Guideline-Directed Medical Therapy for Heart Failure",
            excerpt="ACE inhibitors remain first-line GDMT for HFrEF.",
        )
        generic = _src("Heart failure hospitalization trends")
        gdmt_score = score_merged_source(query, gdmt, anchors=anchors)
        generic_score = score_merged_source(query, generic, anchors=anchors)
        self.assertGreater(gdmt_score.total, generic_score.total)
        self.assertGreater(gdmt_score.features["anchor_excerpt"], 0.0)

    def test_entity_overlap_requires_title_anchor(self) -> None:
        query = "statin primary prevention cardiovascular risk"
        anchors = ("statin", "statins")
        excerpt_only = _src(
            "Cardiovascular Outcomes in Diabetes",
            sid="ek_1",
            excerpt="Statin therapy reduces LDL in primary prevention.",
        )
        title_match = _src(
            "Evaluating simvastatin in multiple sclerosis",
            sid="ek_2",
        )
        excerpt_score = score_merged_source(
            query,
            excerpt_only,
            anchors=anchors,
            query_entity_ids=("entity:drug-class:statins",),
            source_entity_ids=("entity:drug-class:statins",),
        )
        title_score = score_merged_source(
            query,
            title_match,
            anchors=anchors,
            query_entity_ids=("entity:drug-class:statins",),
            source_entity_ids=("entity:drug-class:statins",),
        )
        self.assertEqual(excerpt_score.features["entity"], 0.0)
        self.assertGreater(title_score.features["entity"], 0.0)
        self.assertEqual(title_score.features["anchor_title"], 1.0)
        self.assertEqual(excerpt_score.features["anchor_title"], 0.0)


if __name__ == "__main__":
    unittest.main()
