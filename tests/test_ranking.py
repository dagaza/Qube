"""Tests for Phase 3 ranking modules."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.ranking.diversity import mmr_select_rows  # noqa: E402
from core.knowledge.ranking.freshness import freshness_score  # noqa: E402
from core.knowledge.ranking.relevance import score_rows  # noqa: E402
from core.knowledge.ranking.reliability import apply_reliability_scores  # noqa: E402
from core.knowledge.ranking.stopping import adaptive_stop_reason  # noqa: E402

_PUBMED = {
    "title": "Semaglutide and cardiovascular outcomes",
    "snippet": "Semaglutide reduced major adverse cardiovascular events.",
    "full_text": "Semaglutide reduced major adverse cardiovascular events in adults.",
    "_adapter": "pubmed",
    "_scientific_relevance": 0.82,
}

_OPENALEX = {
    "title": "Cardiovascular effects of GLP-1 agonists",
    "snippet": "GLP-1 receptor agonists including semaglutide show cardiovascular benefit.",
    "full_text": "GLP-1 receptor agonists including semaglutide show cardiovascular benefit.",
    "_adapter": "openalex",
    "_scientific_relevance": 0.75,
}

_TANGENTIAL = {
    "title": "Digital Twins for Healthcare Systems",
    "snippet": "We propose a digital twin framework for hospital operations.",
    "full_text": "We propose a digital twin framework for hospital operations.",
    "_adapter": "arxiv",
    "_scientific_relevance": 0.05,
}


class TestRanking(unittest.TestCase):
    def test_relevance_filters_tangential_hits(self) -> None:
        rows = [_PUBMED, _TANGENTIAL]
        kept, rejected = score_rows(
            rows,
            query="semaglutide cardiovascular outcomes",
            min_score=0.12,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["_adapter"], "pubmed")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["_adapter"], "arxiv")

    def test_mmr_prefers_adapter_diversity(self) -> None:
        dup = dict(_PUBMED)
        dup["title"] = "Semaglutide cardiovascular trial duplicate"
        selected = mmr_select_rows(
            [_PUBMED, dup, _OPENALEX],
            max_results=2,
        )
        adapters = {r["_adapter"] for r in selected}
        self.assertEqual(len(selected), 2)
        self.assertIn("pubmed", adapters)
        self.assertIn("openalex", adapters)

    def test_reliability_scores_applied(self) -> None:
        scored = apply_reliability_scores([_PUBMED, _OPENALEX])
        for row in scored:
            self.assertIn("_reliability_score", row)
            self.assertGreaterEqual(row["_reliability_score"], 0.0)
            self.assertLessEqual(row["_reliability_score"], 0.95)

    def test_freshness_recent_year(self) -> None:
        self.assertEqual(freshness_score("2024-01"), 1.0)

    def test_adaptive_stop_sufficient_evidence(self) -> None:
        reason = adaptive_stop_reason(
            kept_count=3,
            max_results=3,
            avg_relevance=0.45,
            adapter_count=2,
            abstract_count=2,
        )
        self.assertEqual(reason, "sufficient_evidence")


if __name__ == "__main__":
    unittest.main()
