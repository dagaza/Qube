"""Tests for continuous recall-fusion pilot routing layer."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.routing_continuous_pilot import analyze_continuous_pilot_routing, decide_pilot_route
from eval.routing_perturbation import (
    CasePerturbationReport,
    RoutePerturbationAnalysis,
    VariantRunResult,
)
from eval.routing_retrieval_propensity import PropensityThresholds


class PilotRoutingTests(unittest.TestCase):
    def test_decide_pilot_route_low_propensity_is_none(self) -> None:
        route = decide_pilot_route(
            0.2, 0.1, 0.1, thresholds=PropensityThresholds(t_none=0.35), baseline_route="hybrid"
        )
        self.assertEqual(route, "none")

    def test_analyze_pilot_produces_summary(self) -> None:
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
        result = analyze_continuous_pilot_routing(
            RoutePerturbationAnalysis(summary={}, cases=[report])
        )
        self.assertIn("instability_reduction_pct", result.summary)
        self.assertIn("best_thresholds", result.summary)
        self.assertEqual(len(result.variants), 3)
        self.assertEqual(len(result.clusters), 1)


if __name__ == "__main__":
    unittest.main()
