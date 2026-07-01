"""Tests for retrieval eval harness (Phase 6 Slice 1)."""

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

_SCIENTIFIC = Path(_WS_ROOT) / "eval" / "retrieval_corpus" / "v1_scientific.json"
_TRUSTED = Path(_WS_ROOT) / "eval" / "retrieval_corpus" / "v1_trusted.json"


def _load_eval_module():
    path = Path(_WS_ROOT) / "tools" / "evaluate_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestEvaluateRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_eval_module()

    def test_trusted_corpus_schema(self) -> None:
        data = json.loads(_TRUSTED.read_text(encoding="utf-8"))
        self.assertEqual(data["service"], "trusted_knowledge")
        self.assertGreaterEqual(len(data["queries"]), 5)
        for entry in data["queries"]:
            self.assertIn("min_authority", entry)
            self.assertIn("expect_adapters", entry)

    def test_trusted_dry_run(self) -> None:
        _, entries = self.mod._load_corpus(_TRUSTED)
        result = self.mod._evaluate_query(
            entries[0], live=False, knowledge_service="trusted_knowledge"
        )
        self.assertEqual(result["status"], "dry_run")
        self.assertTrue(result.get("trusted_criteria"))

    def test_trusted_live_ok_with_wikipedia(self) -> None:
        entry = {
            "id": "geo",
            "query": "capital of Romania",
            "expect_adapters": ["wikipedia_api"],
            "min_sources": 1,
            "min_authority": 0.9,
            "require_wikipedia": True,
        }
        outcome = self._mock_outcome_wikipedia()
        with patch.object(self.mod, "run_v2_web_retrieval", return_value=outcome):
            result = self.mod._evaluate_query(
                entry, live=True, knowledge_service="trusted_knowledge"
            )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["has_wikipedia"])
        self.assertGreaterEqual(result["max_authority"], 0.9)

    def test_trusted_partial_when_wikipedia_required_missing(self) -> None:
        entry = {
            "id": "geo",
            "query": "capital of Romania",
            "expect_adapters": ["wikipedia_api"],
            "min_sources": 1,
            "min_authority": 0.9,
            "require_wikipedia": True,
        }
        outcome = self._mock_outcome_ddg_only()
        with patch.object(self.mod, "run_v2_web_retrieval", return_value=outcome):
            result = self.mod._evaluate_query(
                entry, live=True, knowledge_service="trusted_knowledge"
            )
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["checks"]["wikipedia_ok"])

    def test_scientific_corpus_has_twelve_discipline_queries(self) -> None:
        data = json.loads(_SCIENTIFIC.read_text(encoding="utf-8"))
        queries = data["queries"]
        self.assertGreaterEqual(len(queries), 12)
        tagged = [
            q for q in queries
            if q.get("discipline") and q.get("primary_adapter")
        ]
        self.assertEqual(len(tagged), 12)
        self.assertEqual(data.get("schema_version"), 2)

    def test_discipline_primary_stats_grouping(self) -> None:
        entries = [
            {"id": "a", "discipline": "biology", "primary_adapter": "pubmed"},
            {"id": "b", "discipline": "biology", "primary_adapter": "pubmed"},
            {"id": "c", "discipline": "chemistry", "primary_adapter": "pubchem"},
        ]
        results = [
            {"checks": {"primary_ok": True}},
            {"checks": {"primary_ok": False}},
            {"checks": {"primary_ok": True}},
        ]
        stats = self.mod._discipline_primary_stats(entries, results)
        self.assertEqual(stats["biology"]["primary_hits"], 1)
        self.assertEqual(stats["biology"]["total"], 2)
        self.assertEqual(stats["biology"]["primary_rate"], 0.5)
        self.assertEqual(
            self.mod._groups_below_threshold(stats, threshold=0.7),
            ["biology"],
        )

    def test_groups_below_threshold_passes_when_all_ok(self) -> None:
        stats = {
            "physics": {"primary_hits": 1, "total": 1, "primary_rate": 1.0},
            "chemistry": {"primary_hits": 1, "total": 1, "primary_rate": 1.0},
        }
        self.assertEqual(
            self.mod._groups_below_threshold(stats, threshold=0.7),
            [],
        )

    def test_scientific_discipline_and_primary_checks(self) -> None:
        from core.knowledge.types import EvidenceBundle, EvidenceObject, WebRetrievalOutcome

        entry = {
            "id": "cs_001",
            "query": "transformer attention mechanism",
            "discipline": "computer_science",
            "primary_adapter": "arxiv",
            "expect_adapters": ["arxiv"],
            "expect_abstract": True,
        }
        src = EvidenceObject(
            id="ek_1",
            source_id="https://arxiv.org/abs/1706.03762",
            adapter="arxiv",
            retrieval_method="abstract",
            title="Attention Is All You Need",
            excerpt="We propose a new architecture.",
            full_text="We propose a new architecture.",
            url="https://arxiv.org/abs/1706.03762",
            document_type="preprint",
            relevance_score=0.85,
            authority_score=0.72,
            reliability_score=0.65,
            fetch_status="abstract",
        )
        bundle = EvidenceBundle(
            bundle_id="b3",
            query_raw=entry["query"],
            query_resolved=entry["query"],
            knowledge_service="scientific_evidence",
            retrieval_strategy="scientific_parallel",
            profile_version="0.1.0",
            retrieved_at=0.0,
            latency_ms=120.0,
            confidence=0.85,
            coverage="adequate",
            coverage_rationale="test",
            authority_summary=0.72,
            reliability_summary=0.65,
            diversity_summary=0.4,
            sources=(src,),
            rejected_count=0,
            warnings=(),
            conflicts=(),
            stop_reason="sufficient_evidence",
            adapter_calls=("arxiv",),
        )
        outcome = WebRetrievalOutcome(
            web_results=[],
            web_results_raw_for_audit=[],
            web_results_kept_for_audit=[],
            relevance_diag=None,
            skip_enrichment=False,
            bundle=bundle,
            latency_ms=120.0,
        )
        with patch.object(self.mod, "run_v2_web_retrieval", return_value=outcome):
            result = self.mod._evaluate_query(
                entry, live=True, knowledge_service="scientific_evidence"
            )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["checks"]["discipline_ok"])
        self.assertTrue(result["checks"]["primary_ok"])
        self.assertEqual(result["detected_discipline"], "computer_science")

    @staticmethod
    def _mock_outcome_wikipedia():
        from core.knowledge.types import EvidenceBundle, EvidenceObject, WebRetrievalOutcome

        src = EvidenceObject(
            id="ek_1",
            source_id="https://en.wikipedia.org/wiki/Bucharest",
            adapter="wikipedia_api",
            retrieval_method="api_extract",
            title="Bucharest",
            excerpt="Capital of Romania",
            full_text="Capital of Romania",
            url="https://en.wikipedia.org/wiki/Bucharest",
            document_type="encyclopedia",
            relevance_score=0.85,
            authority_score=0.95,
            reliability_score=0.8,
            fetch_status="abstract",
        )
        bundle = EvidenceBundle(
            bundle_id="b1",
            query_raw="test",
            query_resolved="test",
            knowledge_service="trusted_knowledge",
            retrieval_strategy="wiki_api_allowlist_ddg",
            profile_version="0.1.0",
            retrieved_at=0.0,
            latency_ms=50.0,
            confidence=0.9,
            coverage="adequate",
            coverage_rationale="test",
            authority_summary=0.95,
            reliability_summary=0.8,
            diversity_summary=0.5,
            sources=(src,),
            rejected_count=0,
            warnings=(),
            conflicts=(),
            stop_reason="sufficient_evidence",
            adapter_calls=("wikipedia_api",),
        )
        return WebRetrievalOutcome(
            web_results=[{"title": "Bucharest", "snippet": "Capital of Romania"}],
            web_results_raw_for_audit=[],
            web_results_kept_for_audit=[],
            relevance_diag=None,
            skip_enrichment=False,
            bundle=bundle,
            latency_ms=50.0,
        )

    @staticmethod
    def _mock_outcome_ddg_only():
        from core.knowledge.types import EvidenceBundle, EvidenceObject, WebRetrievalOutcome

        src = EvidenceObject(
            id="ek_1",
            source_id="https://www.cdc.gov/example",
            adapter="duckduckgo",
            retrieval_method="serp",
            title="CDC page",
            excerpt="Vaccines",
            full_text=None,
            url="https://www.cdc.gov/example",
            document_type="government",
            relevance_score=0.6,
            authority_score=0.88,
            reliability_score=0.5,
            fetch_status="snippet_only",
        )
        bundle = EvidenceBundle(
            bundle_id="b2",
            query_raw="test",
            query_resolved="test",
            knowledge_service="trusted_knowledge",
            retrieval_strategy="wiki_api_allowlist_ddg",
            profile_version="0.1.0",
            retrieved_at=0.0,
            latency_ms=80.0,
            confidence=0.7,
            coverage="adequate",
            coverage_rationale="test",
            authority_summary=0.88,
            reliability_summary=0.5,
            diversity_summary=0.3,
            sources=(src,),
            rejected_count=0,
            warnings=("no_wikipedia_hit",),
            conflicts=(),
            stop_reason="budget_exhausted",
            adapter_calls=("duckduckgo",),
        )
        return WebRetrievalOutcome(
            web_results=[{"title": "CDC page", "snippet": "Vaccines"}],
            web_results_raw_for_audit=[],
            web_results_kept_for_audit=[],
            relevance_diag=None,
            skip_enrichment=False,
            bundle=bundle,
            latency_ms=80.0,
        )


if __name__ == "__main__":
    unittest.main()
