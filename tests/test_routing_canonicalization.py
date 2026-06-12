"""Tests for shadow routing canonicalization learner."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.routing_canonicalization import (
    VariantRecord,
    analyze_routing_canonicalization,
    apply_shadow_boundary,
    canonical_route_majority,
    canonical_route_weighted,
    classify_instability,
)
from eval.routing_perturbation import (
    CasePerturbationReport,
    RoutePerturbationAnalysis,
    VariantRunResult,
)


def _variant(
    vid: str,
    route: str,
    *,
    margin: float = 0.03,
    chat: float = 0.55,
    top: float = 0.55,
    second: float = 0.52,
    hits: int = 1,
    fusion: bool = False,
) -> VariantRecord:
    return VariantRecord(
        variant_id=vid,
        case_id="c1",
        execution_route=route,
        chat_score=chat,
        confidence_margin=margin,
        top_score=top,
        second_best_score=second,
        rag_hits=hits,
        recall_fusion_triggered=fusion,
    )


class CanonicalRouteTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        variants = [
            _variant("v1", "none"),
            _variant("v2", "none"),
            _variant("v3", "hybrid"),
        ]
        self.assertEqual(canonical_route_majority(variants), "none")

    def test_weighted_vote_prefers_high_separation(self) -> None:
        variants = [
            _variant("v1", "none", top=0.55, second=0.54),
            _variant("v2", "hybrid", top=0.80, second=0.40),
        ]
        self.assertEqual(canonical_route_weighted(variants), "hybrid")


class InstabilityClassificationTests(unittest.TestCase):
    def test_purely_ambiguous(self) -> None:
        variants = [
            _variant("v1", "none"),
            _variant("v2", "hybrid"),
            _variant("v3", "rag"),
            _variant("v4", "memory"),
        ]
        canon = canonical_route_majority(variants)
        self.assertEqual(classify_instability(variants, canon), "purely_ambiguous")

    def test_boundary_instability(self) -> None:
        variants = [
            _variant("v1", "none", margin=0.04, top=0.55, second=0.52),
            _variant("v2", "none", margin=0.04, top=0.56, second=0.53),
            _variant("v3", "none", margin=0.04, top=0.54, second=0.51),
            _variant("v4", "hybrid", margin=0.02, top=0.55, second=0.53),
        ]
        canon = canonical_route_majority(variants)
        self.assertEqual(classify_instability(variants, canon), "boundary_instability")

    def test_recall_fusion_instability(self) -> None:
        variants = [
            _variant("v1", "hybrid", fusion=False),
            _variant("v2", "hybrid", fusion=True),
        ]
        canon = canonical_route_majority(variants)
        self.assertEqual(classify_instability(variants, canon), "recall_fusion_instability")


class ShadowBoundaryTests(unittest.TestCase):
    def test_high_chat_forces_none(self) -> None:
        v = _variant("v1", "hybrid", chat=0.85)
        self.assertEqual(apply_shadow_boundary(v, t_chat=0.70, t_margin_low=0.05, t_sep=0.03), "none")

    def test_low_sep_forces_none(self) -> None:
        v = _variant("v1", "hybrid", top=0.52, second=0.50)
        self.assertEqual(apply_shadow_boundary(v, t_chat=0.90, t_margin_low=0.01, t_sep=0.05), "none")


class AnalyzeIntegrationTests(unittest.TestCase):
    def test_analyze_produces_summary(self) -> None:
        report = CasePerturbationReport(
            case_id="gk_rt_002",
            base_prompt="Explain TCP.",
            category="general_knowledge_retrieval_tempting",
            base_route="none",
            variants=[
                VariantRunResult(
                    variant_id="gk_rt_002__v1",
                    text="t1",
                    perturbation_type="paraphrase",
                    route="none",
                    execution_route="none",
                    memory_hits=0,
                    rag_hits=0,
                    web_hits=0,
                    confidence_margin=0.04,
                    top_score=0.55,
                    chat_score=0.6,
                    second_best_score=0.52,
                ),
                VariantRunResult(
                    variant_id="gk_rt_002__v2",
                    text="t2",
                    perturbation_type="paraphrase",
                    route="hybrid",
                    execution_route="hybrid",
                    memory_hits=1,
                    rag_hits=0,
                    web_hits=0,
                    confidence_margin=0.02,
                    top_score=0.56,
                    chat_score=0.58,
                    second_best_score=0.53,
                ),
                VariantRunResult(
                    variant_id="gk_rt_002__v3",
                    text="t3",
                    perturbation_type="paraphrase",
                    route="none",
                    execution_route="none",
                    memory_hits=0,
                    rag_hits=0,
                    web_hits=0,
                    confidence_margin=0.03,
                    top_score=0.54,
                    chat_score=0.62,
                    second_best_score=0.51,
                ),
            ],
            route_consistency_score=0.5,
            retrieval_consistency_score=0.75,
            web_trigger_stability=1.0,
            stability_label="highly_unstable",
            unique_routes=["none", "hybrid"],
            route_variance_pattern="hybrid ↔ none",
            retrieval_variance_pattern="1hits/2miss",
            confidence_margins=[0.04, 0.02, 0.03],
        )
        analysis = analyze_routing_canonicalization(
            RoutePerturbationAnalysis(summary={}, cases=[report])
        )
        self.assertEqual(analysis.summary["clusters_total"], 1)
        self.assertIn("best_threshold_set", analysis.summary)
        self.assertIn("metrics", analysis.summary)
        self.assertGreater(len(analysis.tradeoff_curve), 0)


if __name__ == "__main__":
    unittest.main()
