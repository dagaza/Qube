"""Tests for continuous recall-fusion architectural validation layer."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.routing_arch_validation import analyze_continuous_arch_validation
from eval.routing_perturbation import (
    CasePerturbationReport,
    RoutePerturbationAnalysis,
    VariantRunResult,
)


class ArchValidationTests(unittest.TestCase):
    def _make_analysis(self) -> RoutePerturbationAnalysis:
        variants = [
            VariantRunResult(
                variant_id="c1__v1",
                text="Explain TCP.",
                perturbation_type="paraphrase",
                route="hybrid",
                execution_route="hybrid",
                memory_hits=1,
                rag_hits=0,
                web_hits=0,
                confidence_margin=0.03,
                top_score=0.6,
                chat_score=0.55,
                second_best_score=0.52,
                recall_fusion_triggered=True,
            ),
            VariantRunResult(
                variant_id="c1__v2",
                text="How does TCP work?",
                perturbation_type="paraphrase",
                route="none",
                execution_route="none",
                memory_hits=0,
                rag_hits=0,
                web_hits=0,
                confidence_margin=0.04,
                top_score=0.55,
                chat_score=0.58,
                second_best_score=0.53,
                recall_fusion_triggered=False,
            ),
            VariantRunResult(
                variant_id="c1__v3",
                text="What is TCP?",
                perturbation_type="paraphrase",
                route="hybrid",
                execution_route="hybrid",
                memory_hits=1,
                rag_hits=0,
                web_hits=0,
                confidence_margin=0.02,
                top_score=0.58,
                chat_score=0.56,
                second_best_score=0.51,
                recall_fusion_triggered=True,
            ),
        ]
        report = CasePerturbationReport(
            case_id="gk_rt_002",
            base_prompt="Explain TCP.",
            category="general_knowledge_retrieval_tempting",
            base_route="none",
            variants=variants,
            route_consistency_score=0.5,
            retrieval_consistency_score=0.75,
            web_trigger_stability=1.0,
            stability_label="highly_unstable",
            unique_routes=["none", "hybrid"],
            route_variance_pattern="hybrid ↔ none",
            retrieval_variance_pattern="1hits/2miss",
            confidence_margins=[0.03, 0.04, 0.02],
        )
        return RoutePerturbationAnalysis(summary={}, cases=[report])

    def test_validation_produces_summary_and_sweep(self) -> None:
        result = analyze_continuous_arch_validation(self._make_analysis())
        self.assertIn("instability_reduction_pct", result.summary)
        self.assertIn("by_category", result.summary)
        self.assertIn("flip_type_summary", result.summary)
        self.assertGreater(len(result.threshold_sweep), 0)
        self.assertEqual(len(result.pilot.variants), 3)


if __name__ == "__main__":
    unittest.main()
