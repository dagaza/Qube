"""CI regression gate for offline router evaluation (no embedder required)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_evaluation import (  # noqa: E402
    RouterEvalConfig,
    build_summary,
    compare_runs,
    evaluate_case,
    load_corpus,
    load_run_json,
)
from mcp.cognitive_router import CognitiveRouterV4  # noqa: E402

BASELINE_PATH = (
    Path(ROOT) / "eval" / "runs" / "router_no_embeddings_v1" / "run.json"
)
CORPUS_PATH = Path(ROOT) / "eval" / "router_corpus" / "v1_baseline.json"

# Floors captured Jul 2026 (--no-embeddings, discourse on, no retrieval hits).
MIN_STRICT_ACCURACY = 0.55
MIN_FAMILY_ACCURACY = 0.55


class RouterRegressionGateTests(unittest.TestCase):
    def test_baseline_corpus_routes_without_errors(self) -> None:
        meta, cases = load_corpus(CORPUS_PATH)
        self.assertEqual(meta.get("schema"), "qube.router_corpus.v1")
        router = CognitiveRouterV4()
        config = RouterEvalConfig(install_centroids=False, with_retrieval=False)
        results = [
            evaluate_case(case, router=router, embed_fn=None, config=config)
            for case in cases
        ]
        summary = build_summary(results)
        self.assertFalse(summary.errors)
        self.assertGreaterEqual(summary.strict_accuracy, MIN_STRICT_ACCURACY)
        self.assertGreaterEqual(summary.family_accuracy, MIN_FAMILY_ACCURACY)

    def test_no_regression_vs_committed_baseline(self) -> None:
        if not BASELINE_PATH.is_file():
            self.skipTest(f"missing baseline: {BASELINE_PATH}")

        _meta, cases = load_corpus(CORPUS_PATH)
        router = CognitiveRouterV4()
        config = RouterEvalConfig(install_centroids=False, with_retrieval=False)
        results = [
            evaluate_case(case, router=router, embed_fn=None, config=config)
            for case in cases
        ]
        summary = build_summary(results)
        current = {
            "summary": {
                "execution_final_accuracy": summary.strict_accuracy,
            },
            "results": [
                {
                    "case_id": r.case_id,
                    "execution_final_match": r.execution_final_match,
                    "execution_route_final": r.execution_route_final,
                    "expected_route": r.expected_route,
                }
                for r in results
            ],
        }
        baseline = load_run_json(BASELINE_PATH)
        comparison = compare_runs(baseline, current, min_delta=0.0)
        self.assertFalse(
            comparison.get("regressed"),
            msg=f"router regression: {comparison}",
        )


if __name__ == "__main__":
    unittest.main()
