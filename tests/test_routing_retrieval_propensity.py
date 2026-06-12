"""Tests for shadow continuous retrieval propensity model."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.routing_retrieval_propensity import (
    PropensityThresholds,
    PropensityWeights,
    analyze_retrieval_propensity,
    compute_retrieval_propensity_score,
    compute_retrieval_probabilities,
    decide_shadow_route,
)
from eval.routing_perturbation import (
    CasePerturbationReport,
    RoutePerturbationAnalysis,
    VariantRunResult,
)


class PropensityModelTests(unittest.TestCase):
    def test_low_propensity_forces_none_route(self) -> None:
        thresholds = PropensityThresholds(t_none=0.40)
        p_mem, p_rag, _ = compute_retrieval_probabilities(0.25, memory_affinity=0.7, rag_affinity=0.7)
        route = decide_shadow_route(
            0.25, p_mem, p_rag, thresholds=thresholds, original_route="hybrid"
        )
        self.assertEqual(route, "none")

    def test_strong_separation_increases_propensity(self) -> None:
        low = compute_retrieval_propensity_score(
            top_score=0.52,
            second_best_score=0.50,
            chat_score=0.7,
            confidence_margin=0.1,
            follow_up_boost=0.0,
            discourse_signal=0.0,
            weights=PropensityWeights(),
        )
        high = compute_retrieval_propensity_score(
            top_score=0.85,
            second_best_score=0.30,
            chat_score=0.7,
            confidence_margin=0.1,
            follow_up_boost=0.0,
            discourse_signal=0.0,
            weights=PropensityWeights(),
        )
        self.assertGreater(high, low)


class AnalyzePropensityTests(unittest.TestCase):
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

    def test_analyze_produces_summary(self) -> None:
        result = analyze_retrieval_propensity(self._make_analysis())
        self.assertIn("avg_propensity_score", result.summary)
        self.assertIn("best_thresholds", result.summary)
        self.assertEqual(len(result.variants), 3)
        self.assertEqual(len(result.clusters), 1)


if __name__ == "__main__":
    unittest.main()
