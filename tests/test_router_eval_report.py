"""Tests for router evaluation report formatting."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_eval_report import format_evaluation_report
from core.router_evaluation import RouterEvalSummary


class ReportFormatTests(unittest.TestCase):
    def test_report_contains_key_sections(self) -> None:
        summary = RouterEvalSummary(
            total=2,
            router_accuracy=0.5,
            execution_pre_accuracy=0.5,
            execution_final_accuracy=0.5,
            strict_accuracy=0.5,
            family_accuracy=1.0,
            downgrade_count=1,
            downgrade_rate=0.5,
            rewrite_applied_count=0,
            failure_causes={"no_failure": 1, "recall_fusion_upgrade": 1},
            rewrite_impact={"attempted_count": 0},
            memory_analysis={"total": 0},
            by_expected_route={},
            by_category={
                "rag_retrieval": {
                    "total": 2,
                    "strict_accuracy": 0.5,
                    "family_accuracy": 1.0,
                    "downgrade_count": 1,
                }
            },
            confusion_matrix={"rag": {"hybrid": 1, "rag": 1}},
            retrieval_hit_rates={"rag_retrieval": 1.0},
            retrieval_calibration={
                "over_retrieval_rate": 0.1,
                "over_retrieval_count": 5,
                "chat_labeled_total": 50,
                "under_retrieval_rate": 0.05,
                "retrieval_necessity_error_count": 2,
                "retrieval_expected_total": 40,
                "recall_fusion_over_retrieval_share": 0.6,
                "recall_fusion_over_retrieval_count": 3,
                "avg_chat_score_correct_chat_cases": 0.72,
                "avg_chat_score_over_retrieval_cases": 0.55,
                "potential_chat_guard_threshold_candidate": 0.67,
                "chat_margin_histogram": {"0-0.05": 10, "0.20+": 5},
                "over_retrieval_by_category": {
                    "general_knowledge_retrieval_tempting": {
                        "over_retrieval_rate": 0.2,
                        "over_retrieval_count": 4,
                        "chat_labeled_total": 20,
                    }
                },
                "retrieval_suppression_candidates": [
                    {
                        "case_id": "gk_rt_001",
                        "prompt": "Tell me about Linux.",
                        "chat_score": 0.81,
                        "top_intent": "recall",
                        "route_taken": "hybrid",
                        "retrieval_type": "rag",
                        "retrieval_hits": 2,
                        "recall_fusion_triggered": True,
                        "confidence_margin": 0.04,
                    }
                ],
            },
            errors=[],
        )
        text = format_evaluation_report(summary)
        self.assertIn("Strict accuracy", text)
        self.assertIn("Retrieval Calibration Summary", text)
        self.assertIn("Over-retrieval rate", text)
        self.assertIn("suppression candidates", text)
        self.assertIn("gk_rt_001", text)


if __name__ == "__main__":
    unittest.main()
