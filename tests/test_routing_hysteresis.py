"""Tests for shadow routing hysteresis simulation."""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.routing_hysteresis import (
    HysteresisConfig,
    apply_hysteresis_shadow_route,
    simulate_hysteresis_on_perturbation,
)
from eval.routing_perturbation import (
    CasePerturbationReport,
    RoutePerturbationAnalysis,
    VariantRunResult,
)


class ApplyHysteresisTests(unittest.TestCase):
    def test_rule_a_blocks_weak_retrieval_from_none_anchor(self) -> None:
        route = apply_hysteresis_shadow_route(
            "hybrid",
            confidence_margin=0.08,
            chat_score=0.55,
            top_score=0.52,
            second_best_score=0.50,
            previous_route="none",
        )
        self.assertEqual(route, "none")

    def test_rule_b_keeps_retrieval_on_low_margin(self) -> None:
        route = apply_hysteresis_shadow_route(
            "rag",
            confidence_margin=0.03,
            chat_score=0.40,
            top_score=0.70,
            second_best_score=0.50,
            previous_route="rag",
        )
        self.assertEqual(route, "rag")

    def test_rule_c_locks_low_margin_to_majority(self) -> None:
        route = apply_hysteresis_shadow_route(
            "hybrid",
            confidence_margin=0.02,
            chat_score=0.60,
            top_score=0.65,
            second_best_score=0.40,
            previous_route="none",
        )
        self.assertEqual(route, "none")


@dataclass
class _FakePerturbation:
    cases: list


class SimulateHysteresisTests(unittest.TestCase):
    def _variant(
        self,
        vid: str,
        route: str,
        *,
        margin: float = 0.03,
        gap_top: float = 0.55,
        gap_second: float = 0.52,
    ) -> VariantRunResult:
        return VariantRunResult(
            variant_id=vid,
            text="t",
            perturbation_type="paraphrase",
            route=route,
            execution_route=route,
            memory_hits=1,
            rag_hits=0,
            web_hits=0,
            confidence_margin=margin,
            top_score=gap_top,
            chat_score=0.6,
            second_best_score=gap_second,
        )

    def test_simulation_reduces_hybrid_none_flips(self) -> None:
        report = CasePerturbationReport(
            case_id="gk_rt_002",
            base_prompt="Explain TCP.",
            category="general_knowledge_retrieval_tempting",
            base_route="none",
            variants=[
                self._variant("v1", "none", margin=0.04),
                self._variant("v2", "hybrid", margin=0.02),
                self._variant("v3", "hybrid", margin=0.03),
                self._variant("v4", "none", margin=0.04),
            ],
            route_consistency_score=0.5,
            retrieval_consistency_score=0.9,
            web_trigger_stability=1.0,
            stability_label="highly_unstable",
            unique_routes=["none", "hybrid"],
            route_variance_pattern="hybrid ↔ none",
            retrieval_variance_pattern="hits",
            confidence_margins=[0.04, 0.02, 0.03, 0.04],
        )
        analysis = RoutePerturbationAnalysis(summary={}, cases=[report])
        result = simulate_hysteresis_on_perturbation(analysis)
        self.assertGreater(result.summary["stability_gain"], 0.0)
        self.assertGreaterEqual(result.summary["hybrid_none_prevented"], 0)

    def test_retrieval_consistency_unchanged(self) -> None:
        report = CasePerturbationReport(
            case_id="mem_001",
            base_prompt="What is my dog's name?",
            category="memory_recall",
            base_route="hybrid",
            variants=[
                self._variant("v1", "hybrid", margin=0.15, gap_top=0.8, gap_second=0.4),
                self._variant("v2", "hybrid", margin=0.12, gap_top=0.75, gap_second=0.35),
                self._variant("v3", "memory", margin=0.20, gap_top=0.85, gap_second=0.30),
            ],
            route_consistency_score=0.67,
            retrieval_consistency_score=1.0,
            web_trigger_stability=1.0,
            stability_label="moderately_unstable",
            unique_routes=["hybrid", "memory"],
            route_variance_pattern="hybrid ↔ memory",
            retrieval_variance_pattern="hits",
            confidence_margins=[0.15, 0.12, 0.20],
        )
        analysis = RoutePerturbationAnalysis(summary={}, cases=[report])
        result = simulate_hysteresis_on_perturbation(analysis)
        self.assertAlmostEqual(result.summary["retrieval_consistency_delta"], 0.0, places=4)
        self.assertFalse(result.summary["safety_flag"])


if __name__ == "__main__":
    unittest.main()
