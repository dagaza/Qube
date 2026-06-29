"""Tests for Stage 1 scientific query planner (QA-3B/3C failure cases)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.pipeline_scientific import ScientificEvidencePipeline  # noqa: E402
from core.knowledge.scientific_query_planner import (  # noqa: E402
    adapter_query_for,
    plan_scientific_query,
)
from core.knowledge.types import RetrievalBudget, RetrievalContext  # noqa: E402

_QA_3B = (
    "Summarize key outcomes from the EMPEROR-Reduced trial for heart failure."
)
_QA_3C = "What does the literature say about dapagliflozin in HFrEF?"
_QA_3A = "What is the evidence for ACE inhibitors in heart failure?"


class TestScientificQueryPlanner(unittest.TestCase):
    def test_qa_3b_emperor_keyword_plan(self) -> None:
        plan = plan_scientific_query(_QA_3B)
        self.assertEqual(
            plan.semantic_query,
            "Summarize key outcomes from the EMPEROR-Reduced trial for heart failure",
        )
        self.assertIn("EMPEROR-Reduced", plan.keyword_query)
        self.assertIn("heart failure", plan.keyword_query.lower())
        self.assertIn("emperor-reduced", plan.entity_keywords[0].lower())

    def test_qa_3c_dapagliflozin_keyword_plan(self) -> None:
        plan = plan_scientific_query(_QA_3C)
        self.assertEqual(
            plan.semantic_query,
            "What does the literature say about dapagliflozin in HFrEF",
        )
        self.assertIn("dapagliflozin", plan.keyword_query.lower())
        self.assertIn("HFrEF", plan.keyword_query)

    def test_qa_3a_semantic_unchanged_keyword_still_usable(self) -> None:
        plan = plan_scientific_query(_QA_3A)
        self.assertEqual(
            plan.semantic_query,
            "What is the evidence for ACE inhibitors in heart failure",
        )
        self.assertIn("ACE inhibitors", plan.keyword_query)
        self.assertIn("heart failure", plan.keyword_query.lower())

    def test_adapter_query_routing(self) -> None:
        plan = plan_scientific_query(_QA_3B)
        self.assertNotEqual(plan.keyword_query, plan.semantic_query)
        self.assertEqual(adapter_query_for(plan, "pubmed"), plan.keyword_query)
        self.assertEqual(adapter_query_for(plan, "arxiv"), plan.keyword_query)
        self.assertEqual(adapter_query_for(plan, "openalex"), plan.semantic_query)

    def test_non_medical_query_skips_entity_keywords(self) -> None:
        plan = plan_scientific_query(
            "transformer attention mechanism neural machine translation"
        )
        self.assertEqual(plan.entity_keywords, ())
        self.assertIn("transformer", plan.keyword_query.lower())


class TestScientificPipelineAdapterSelection(unittest.TestCase):
    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_non_medical_query_skips_pubmed(self, _set_cache, _get_cache) -> None:
        pubmed_calls: list[str] = []

        def _pubmed(q: str, max_results: int = 3) -> list[dict]:
            pubmed_calls.append(q)
            return []

        with patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _pubmed,
                "openalex": lambda q, max_results=3: [
                    {
                        "title": "Attention Is All You Need",
                        "snippet": "Transformer architecture.",
                        "full_text": "We propose a new architecture.",
                        "url": "https://openalex.org/w1",
                        "_adapter": "openalex",
                    }
                ],
                "arxiv": lambda q, max_results=3: [],
            },
        ):
            pipeline = ScientificEvidencePipeline()
            _bundle, rel_diag, _ = pipeline.run(
                RetrievalContext(
                    query="transformer attention mechanism neural machine translation",
                    semantic_query="transformer attention mechanism neural machine translation",
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(pubmed_calls, [])
        assert rel_diag is not None
        self.assertEqual(rel_diag["scientific_adapters_selected"], ["openalex", "arxiv"])


class TestScientificPipelinePlannerIntegration(unittest.TestCase):
    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_pubmed_receives_keyword_query_for_qa_3b(self, _set_cache, _get_cache) -> None:
        pubmed_queries: list[str] = []

        def _pubmed(q: str, max_results: int = 3) -> list[dict]:
            pubmed_queries.append(q)
            return [
                {
                    "title": "Empagliflozin in Heart Failure with a Preserved Ejection Fraction",
                    "snippet": "EMPEROR-Reduced trial outcomes.",
                    "full_text": "EMPEROR-Reduced trial outcomes for heart failure.",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                    "_adapter": "pubmed",
                    "document_type": "journal_abstract",
                }
            ]

        with patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _pubmed,
                "openalex": lambda q, max_results=3: [],
                "arxiv": lambda q, max_results=3: [],
            },
        ):
            pipeline = ScientificEvidencePipeline()
            bundle, rel_diag, _ = pipeline.run(
                RetrievalContext(
                    query=_QA_3B,
                    semantic_query=_QA_3B,
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(len(pubmed_queries), 1)
        self.assertIn("EMPEROR-Reduced", pubmed_queries[0])
        self.assertNotIn("Summarize key outcomes", pubmed_queries[0])
        assert rel_diag is not None
        self.assertIn("EMPEROR-Reduced", rel_diag["scientific_keyword_query"])
        self.assertIn("pubmed", bundle.adapter_calls)

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_pubmed_receives_keyword_query_for_qa_3c(self, _set_cache, _get_cache) -> None:
        pubmed_queries: list[str] = []

        def _pubmed(q: str, max_results: int = 3) -> list[dict]:
            pubmed_queries.append(q)
            return [
                {
                    "title": "Dapagliflozin in heart failure with reduced ejection fraction",
                    "snippet": "DAPA-HF outcomes.",
                    "full_text": "Dapagliflozin reduced heart failure events in HFrEF.",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/2/",
                    "_adapter": "pubmed",
                    "document_type": "journal_abstract",
                }
            ]

        with patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _pubmed,
                "openalex": lambda q, max_results=3: [],
                "arxiv": lambda q, max_results=3: [],
            },
        ):
            pipeline = ScientificEvidencePipeline()
            bundle, rel_diag, _ = pipeline.run(
                RetrievalContext(
                    query=_QA_3C,
                    semantic_query=_QA_3C,
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(len(pubmed_queries), 1)
        self.assertIn("dapagliflozin", pubmed_queries[0].lower())
        self.assertNotIn("What does the literature", pubmed_queries[0])
        assert rel_diag is not None
        self.assertIn("dapagliflozin", rel_diag["scientific_keyword_query"].lower())
        self.assertIn("pubmed", bundle.adapter_calls)


if __name__ == "__main__":
    unittest.main()
