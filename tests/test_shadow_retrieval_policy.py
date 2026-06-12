"""Tests for shadow retrieval policy (LLMWorker observational layer)."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.shadow_retrieval_policy import (
    PolicyThresholds,
    PolicyWeights,
    ShadowRetrievalPolicyTelemetry,
    ShadowRetrievalState,
    axes_activate_retrieval,
    compute_retrieval_policy,
    decompose_propensity_axes,
    shadow_retrieval_policy_enabled,
)
from eval.routing_perturbation import (
    CasePerturbationReport,
    RoutePerturbationAnalysis,
    VariantRunResult,
)
from eval.shadow_retrieval_policy_eval import analyze_shadow_retrieval_policy
from core.router_eval_report import _format_shadow_retrieval_policy


class ComputeRetrievalPolicyTests(unittest.TestCase):
    def test_returns_required_keys(self) -> None:
        state = ShadowRetrievalState(
            baseline_route="HYBRID",
            decision={
                "chat_score": 0.4,
                "confidence_margin": 0.15,
                "top_score": 0.7,
                "second_best_score": 0.45,
            },
            prompt="What did we discuss about TCP?",
            follow_up_strength=0.8,
            discourse_continuation=0.5,
        )
        policy = compute_retrieval_policy(state)
        self.assertIn("retrieval_propensity_score", policy)
        self.assertIn("P_memory", policy)
        self.assertIn("P_rag", policy)
        self.assertIn("P_hybrid", policy)
        self.assertIn("shadow_decision", policy)
        self.assertIn("baseline_recall_fusion", policy)
        self.assertIn("delta_vs_baseline", policy)
        self.assertIn("route_change", policy["delta_vs_baseline"])
        self.assertIn("retrieval_change", policy["delta_vs_baseline"])

    def test_low_propensity_shadow_none(self) -> None:
        state = ShadowRetrievalState(
            baseline_route="hybrid",
            decision={},
            chat_score=0.95,
            confidence_margin=0.9,
            top_score=0.51,
            second_best_score=0.50,
            thresholds=PolicyThresholds(t_none=0.30),
        )
        policy = compute_retrieval_policy(state)
        self.assertEqual(policy["shadow_decision"], "none")
        self.assertLess(policy["retrieval_propensity_score"], 0.30)

    def test_high_propensity_prefers_memory_on_hybrid_baseline(self) -> None:
        state = ShadowRetrievalState(
            baseline_route="hybrid",
            decision={},
            chat_score=0.2,
            confidence_margin=0.05,
            top_score=0.9,
            second_best_score=0.2,
            follow_up_strength=0.9,
            discourse_continuation=0.7,
            baseline_recall_fusion=True,
        )
        policy = compute_retrieval_policy(state)
        self.assertGreater(policy["retrieval_propensity_score"], 0.5)
        self.assertIn(policy["shadow_decision"], {"memory", "rag", "hybrid"})
        self.assertTrue(policy["baseline_recall_fusion"])


class ShadowTelemetryTests(unittest.TestCase):
    def test_summarize_metrics(self) -> None:
        tel = ShadowRetrievalPolicyTelemetry()
        policy_hybrid = compute_retrieval_policy(
            ShadowRetrievalState(
                baseline_route="hybrid",
                decision={"recall_fusion": True},
                chat_score=0.3,
                confidence_margin=0.1,
                top_score=0.8,
                second_best_score=0.3,
                baseline_recall_fusion=True,
            )
        )
        tel.record(
            baseline_route="hybrid",
            shadow_policy=policy_hybrid,
            prompt="recall test",
        )
        policy_none = compute_retrieval_policy(
            ShadowRetrievalState(
                baseline_route="none",
                decision={},
                chat_score=0.9,
                confidence_margin=0.8,
                top_score=0.52,
                second_best_score=0.51,
            )
        )
        tel.record(
            baseline_route="none",
            shadow_policy=policy_none,
            prompt="chat test",
        )
        summary = tel.summarize()
        self.assertEqual(summary["samples"], 2)
        self.assertIn("divergence_rate", summary)
        self.assertIn("recall_fusion_eliminated_rate", summary)
        self.assertIn("hybrid_reduction_rate", summary)
        self.assertIn("retrieval_stability_gain_estimate", summary)
        self.assertIn("best_thresholds", summary)
        self.assertEqual(summary["best_thresholds"]["T_none"], 0.30)


class ShadowEvalAnalysisTests(unittest.TestCase):
    def _fusion_flip_analysis(self) -> RoutePerturbationAnalysis:
        variants = [
            VariantRunResult(
                variant_id="c1__v1",
                text="What did we say about TCP?",
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
                text="Explain TCP.",
                perturbation_type="paraphrase",
                route="none",
                execution_route="none",
                memory_hits=0,
                rag_hits=0,
                web_hits=0,
                confidence_margin=0.12,
                top_score=0.55,
                chat_score=0.7,
                second_best_score=0.50,
                recall_fusion_triggered=False,
            ),
        ]
        case = CasePerturbationReport(
            case_id="c1",
            base_prompt="Explain TCP.",
            category="follow_up",
            base_route="hybrid",
            variants=variants,
            route_consistency_score=0.5,
            retrieval_consistency_score=0.5,
            web_trigger_stability=1.0,
            stability_label="highly_unstable",
            unique_routes=["hybrid", "none"],
            route_variance_pattern="hybrid/none",
            retrieval_variance_pattern="1hits/1miss",
            confidence_margins=[0.03, 0.12],
        )
        return RoutePerturbationAnalysis(summary={}, cases=[case])

    def test_analyze_shadow_on_perturbation(self) -> None:
        analysis = analyze_shadow_retrieval_policy(self._fusion_flip_analysis())
        summary = analysis.summary
        self.assertIn("avg_propensity_score", summary)
        self.assertIn("divergence_rate", summary)
        self.assertIn("recall_fusion_eliminated_rate", summary)
        self.assertIn("hybrid_stability_gain", summary)
        self.assertIn("retrieval_coverage_delta", summary)
        self.assertIn("interpretation", summary)
        self.assertEqual(len(analysis.variant_records), 2)

    def test_report_section_renders(self) -> None:
        analysis = analyze_shadow_retrieval_policy(self._fusion_flip_analysis())
        text = _format_shadow_retrieval_policy(analysis.summary)
        self.assertIn("Avg propensity score", text)
        self.assertIn("Recall-fusion eliminated rate", text)


class AxisDecompositionTests(unittest.TestCase):
    def test_axes_sum_to_combined_propensity(self) -> None:
        state = ShadowRetrievalState(
            baseline_route="hybrid",
            decision={},
            chat_score=0.4,
            confidence_margin=0.1,
            top_score=0.7,
            second_best_score=0.4,
            follow_up_strength=0.8,
            discourse_continuation=0.5,
        )
        axes = decompose_propensity_axes(state)
        self.assertAlmostEqual(
            axes.combined, axes.semantic_raw + axes.contextual_raw, places=4
        )
        policy = compute_retrieval_policy(state)
        self.assertIn("semantic_axis_score", policy)
        self.assertIn("contextual_axis_score", policy)

    def test_2d_or_gate_contextual_only(self) -> None:
        state = ShadowRetrievalState(
            baseline_route="none",
            decision={},
            chat_score=0.95,
            confidence_margin=0.9,
            top_score=0.51,
            second_best_score=0.50,
            follow_up_strength=0.9,
            discourse_continuation=0.7,
        )
        axes = decompose_propensity_axes(state)
        self.assertTrue(axes_activate_retrieval(axes, t_semantic=0.0, t_contextual=0.5))
        self.assertFalse(axes_activate_retrieval(axes, t_semantic=0.0, t_contextual=0.95))

        semantic_axes = decompose_propensity_axes(
            ShadowRetrievalState(
                baseline_route="hybrid",
                decision={},
                chat_score=0.2,
                confidence_margin=0.05,
                top_score=0.9,
                second_best_score=0.2,
            )
        )
        self.assertTrue(
            axes_activate_retrieval(semantic_axes, t_semantic=0.5, t_contextual=0.0)
        )

    def test_2d_thresholds_in_policy(self) -> None:
        state = ShadowRetrievalState(
            baseline_route="hybrid",
            decision={},
            chat_score=0.5,
            confidence_margin=0.1,
            top_score=0.7,
            second_best_score=0.4,
            thresholds=PolicyThresholds(t_semantic=0.9, t_contextual=0.9),
        )
        policy = compute_retrieval_policy(state)
        self.assertEqual(policy["shadow_decision"], "none")


class ShadowPolicyEnvTests(unittest.TestCase):
    def test_enabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUBE_SHADOW_RETRIEVAL_POLICY", None)
            self.assertTrue(shadow_retrieval_policy_enabled())

    def test_disabled_via_env(self) -> None:
        with mock.patch.dict(os.environ, {"QUBE_SHADOW_RETRIEVAL_POLICY": "0"}):
            self.assertFalse(shadow_retrieval_policy_enabled())


if __name__ == "__main__":
    unittest.main()
