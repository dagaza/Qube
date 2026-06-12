"""Tests for route perturbation invariance harness."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_evaluation import RouterEvalCase
from eval.routing_perturbation import (
    VariantRunResult,
    _retrieval_consistency,
    _route_consistency,
    _stability_label,
    _web_trigger_stability,
    generate_perturbations,
)


class GeneratePerturbationsTests(unittest.TestCase):
    def test_explain_tcp_produces_variants(self) -> None:
        case = RouterEvalCase(
            id="gk_rt_002",
            prompt="Explain TCP.",
            expected_route="none",
            category="general_knowledge_retrieval_tempting",
        )
        variants = generate_perturbations(case)
        self.assertGreaterEqual(len(variants), 3)
        texts = [v["text"].lower() for v in variants]
        self.assertTrue(any("how does tcp" in t for t in texts))

    def test_tell_me_about_paraphrase(self) -> None:
        case = RouterEvalCase(
            id="gk_rt_001",
            prompt="Tell me about Linux.",
            expected_route="none",
            category="general_knowledge_retrieval_tempting",
        )
        variants = generate_perturbations(case)
        texts = " ".join(v["text"].lower() for v in variants)
        self.assertIn("what is linux", texts)

    def test_follow_up_deixis_with_history(self) -> None:
        case = RouterEvalCase(
            id="fu_001",
            prompt="What else should I know?",
            expected_route="none",
            category="follow_up",
            history=(
                {"role": "user", "content": "Tell me about Kubernetes."},
                {"role": "assistant", "content": "Kubernetes is a container orchestrator."},
            ),
        )
        variants = generate_perturbations(case)
        types = {v["perturbation_type"] for v in variants}
        self.assertIn("deixis", types)


class ConsistencyScoringTests(unittest.TestCase):
    def _var(self, route: str, hits: int = 0) -> VariantRunResult:
        return VariantRunResult(
            variant_id="v",
            text="t",
            perturbation_type="paraphrase",
            route=route,
            execution_route=route,
            memory_hits=hits,
            rag_hits=0,
            web_hits=0,
            confidence_margin=0.1,
            top_score=0.5,
            chat_score=0.6,
        )

    def test_route_consistency_all_same(self) -> None:
        vars_ = [self._var("none"), self._var("none"), self._var("none")]
        # 1 - (unique_routes / total) = 1 - 1/3
        self.assertAlmostEqual(_route_consistency(vars_), 2 / 3)

    def test_route_consistency_split(self) -> None:
        vars_ = [self._var("none"), self._var("hybrid"), self._var("rag")]
        self.assertAlmostEqual(_route_consistency(vars_), 0.0)

    def test_retrieval_consistency_binary_split(self) -> None:
        vars_ = [self._var("hybrid", 2), self._var("hybrid", 0), self._var("hybrid", 0)]
        score = _retrieval_consistency(vars_)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.5)

    def test_stability_labels(self) -> None:
        self.assertEqual(_stability_label(0.9), "stable")
        self.assertEqual(_stability_label(0.7), "moderately_unstable")
        self.assertEqual(_stability_label(0.4), "highly_unstable")

    def test_web_trigger_stability(self) -> None:
        vars_ = [self._var("web"), self._var("none")]
        self.assertLess(_web_trigger_stability(vars_), 1.0)


if __name__ == "__main__":
    unittest.main()
