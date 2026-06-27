"""Tests for deep-research eval harness (Phase 5 slice 1)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

_CORPUS = Path(_WS_ROOT) / "eval" / "retrieval_corpus" / "v1_deep_research.json"


def _load_eval_module():
    path = Path(_WS_ROOT) / "tools" / "evaluate_deep_research.py"
    spec = importlib.util.spec_from_file_location("evaluate_deep_research", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestEvaluateDeepResearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_eval_module()

    def test_corpus_has_relevance_fields(self) -> None:
        data = json.loads(_CORPUS.read_text(encoding="utf-8"))
        for entry in data["queries"]:
            self.assertTrue(entry.get("expect_any_tokens"))
            self.assertIn("reject_title_patterns", entry)
            self.assertIn("min_relevant_in_top", entry)

    def test_dry_run_reports_relevance_criteria(self) -> None:
        entries = self.mod._load_corpus(_CORPUS)
        result = self.mod._evaluate_query(entries[0], live=False)
        self.assertEqual(result["status"], "dry_run")
        self.assertTrue(result.get("relevance_criteria"))

    def test_live_eval_includes_relevance_ok(self) -> None:
        from core.knowledge.deep_research import DeepResearchResult

        mock_result = DeepResearchResult(
            query="ACE inhibitors heart failure mortality evidence",
            sub_queries=("ACE inhibitors heart failure mortality evidence",),
            merged_bundle=self._mock_bundle(),
            latency_ms=100.0,
            diagnostics={
                "merged_relevance_dropped": 1,
                "merged_sources_pre_filter": 4,
                "merged_sources_post_filter": 3,
            },
        )
        entry = {
            "id": "ace",
            "query": "ACE inhibitors heart failure mortality evidence",
            "expect_adapters": ["pubmed"],
            "min_merged_sources": 2,
            "min_coverage_rank": "adequate",
            "expect_any_tokens": ["ace", "angiotensin"],
            "reject_title_patterns": ["takotsubo"],
            "relevance_top_n": 3,
            "min_relevant_in_top": 2,
        }
        with patch.object(self.mod, "run_deep_research", return_value=mock_result):
            result = self.mod._evaluate_query(entry, live=True)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["relevance_ok"])
        self.assertEqual(result["diagnostics"]["merged_relevance_dropped"], 1)

    @staticmethod
    def _mock_bundle():
        from core.knowledge.types import EvidenceBundle, EvidenceObject

        sources = tuple(
            EvidenceObject(
                id=f"ek_{i}",
                source_id=f"ek_{i}",
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
            for i, title in enumerate(
                (
                    "ACE inhibitors reduce mortality in heart failure",
                    "Angiotensin converting enzyme inhibitors in HF",
                    "Heart failure event rates in trials",
                ),
                start=1,
            )
        )
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
            coverage="excellent",
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


if __name__ == "__main__":
    unittest.main()
