"""Tests for post-hoc routing stability analysis."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_evaluation import RouterEvalResult
from eval.routing_stability import (
    analyze_routing_stability,
    annotate_results_with_stability,
    build_clusters,
    compute_cluster_metrics,
    detect_oscillation,
    export_stability_clusters_json,
    token_jaccard_similarity,
)


def _result(
    case_id: str,
    prompt: str,
    route: str,
    *,
    margin: float = 0.15,
    recall_fusion: bool = False,
) -> RouterEvalResult:
    return RouterEvalResult(
        case_id=case_id,
        prompt=prompt,
        expected_route="none",
        category="general_knowledge",
        notes="",
        router_route=route,
        execution_route_pre_retrieval=route,
        execution_route_final=route,
        top_intent="recall",
        top_score=0.8,
        chat_score=0.7,
        confidence_margin=margin,
        memory_hits=0,
        rag_hits=0,
        web_hits=0,
        downgrade_fired=False,
        rewrite_applied=False,
        router_match=False,
        execution_pre_match=False,
        execution_final_match=False,
        recall_fusion_triggered=recall_fusion,
    )


class TokenSimilarityTests(unittest.TestCase):
    def test_jaccard_identical(self) -> None:
        self.assertEqual(
            token_jaccard_similarity("explain tcp networking", "explain tcp networking"),
            1.0,
        )

    def test_jaccard_partial(self) -> None:
        sim = token_jaccard_similarity("explain tcp protocol", "explain tcp stack")
        self.assertGreater(sim, 0.3)
        self.assertLess(sim, 1.0)


class ClusterMetricsTests(unittest.TestCase):
    def test_instability_and_entropy(self) -> None:
        members = [
            _result("a", "explain tcp", "none"),
            _result("b", "explain tcp protocol", "hybrid"),
        ]
        stats = compute_cluster_metrics("stab_0000", members)
        self.assertEqual(stats.dominant_route, "none")
        self.assertAlmostEqual(stats.instability_score, 0.5)
        self.assertGreater(stats.entropy, 0.0)

    def test_detect_hybrid_none_oscillation(self) -> None:
        members = [
            _result("a", "tell me about linux", "none", margin=0.2),
            _result("b", "tell me about linux kernel", "hybrid", margin=0.12),
        ]
        is_osc, reason = detect_oscillation(members)
        self.assertTrue(is_osc)
        self.assertEqual(reason, "hybrid_vs_none_flip")

    def test_no_oscillation_low_margin(self) -> None:
        members = [
            _result("a", "x", "none", margin=0.05),
            _result("b", "y", "hybrid", margin=0.08),
        ]
        self.assertFalse(detect_oscillation(members)[0])


class StabilityAnalysisTests(unittest.TestCase):
    def test_build_clusters_token_fallback(self) -> None:
        results = [
            _result("a", "explain tcp protocol", "none"),
            _result("b", "explain tcp stack", "hybrid"),
            _result("c", "completely unrelated zebra pizza", "rag"),
        ]
        groups, method = build_clusters(results, embed_fn=None, threshold=0.5)
        self.assertEqual(method, "token_jaccard")
        self.assertGreaterEqual(len(groups), 2)

    def test_annotate_results(self) -> None:
        results = [
            _result("gk_rt_001", "tell me about linux", "hybrid"),
            _result("gk_rt_002", "tell me about linux kernel", "none"),
        ]
        analysis = analyze_routing_stability(results, embed_fn=None, threshold=0.5)
        annotated = annotate_results_with_stability(results, analysis)
        self.assertTrue(all(r.stability_cluster_id for r in annotated))
        self.assertGreater(annotated[0].stability_cluster_size, 0)

    def test_export_json(self) -> None:
        results = [_result("a", "explain tcp", "none")]
        analysis = analyze_routing_stability(results, embed_fn=None)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clusters.json"
            export_stability_clusters_json(path, analysis)
            self.assertTrue(path.is_file())
            self.assertIn("clusters", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
