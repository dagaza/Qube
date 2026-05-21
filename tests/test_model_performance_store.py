from __future__ import annotations

import tempfile
import unittest

from core.output_validation import OutputValidationResult
from core.model_performance_store import ModelPerformanceStore


class TestModelPerformanceStore(unittest.TestCase):
    def test_update_correctness_from_known_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ModelPerformanceStore(path=f"{td}/perf.json")
            v_ok = OutputValidationResult(is_valid=True, issues=[], severity="low")
            v_bad = OutputValidationResult(
                is_valid=False, issues=["template_leakage"], severity="high"
            )

            r1 = store.update_model_metrics("m1", v_ok, 0.8, 2.0, retry_used=False)
            assert r1 is not None
            self.assertEqual(r1.total_requests, 1)
            self.assertEqual(r1.successful_outputs, 1)
            self.assertEqual(r1.structural_failure_rate, 0.0)

            r2 = store.update_model_metrics("m1", v_bad, 0.2, 6.0, retry_used=True)
            assert r2 is not None
            self.assertEqual(r2.total_requests, 2)
            self.assertEqual(r2.successful_outputs, 1)
            self.assertGreater(r2.structural_failure_rate, 0.0)
            self.assertLessEqual(r2.structural_failure_rate, 1.0)
            self.assertGreaterEqual(r2.avg_response_quality, 0.0)
            self.assertLessEqual(r2.avg_response_quality, 1.0)
            self.assertGreater(r2.avg_latency, 0.0)

    def test_stability_repeated_updates_do_not_drift_wildly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ModelPerformanceStore(path=f"{td}/perf.json")
            v_ok = OutputValidationResult(is_valid=True, issues=[], severity="low")
            for _ in range(100):
                store.update_model_metrics("stable", v_ok, 0.7, 2.5, retry_used=False)
            rec = store.get("stable")
            assert rec is not None
            self.assertLess(abs(rec.avg_response_quality - 0.7), 0.05)
            self.assertLess(abs(rec.avg_latency - 2.5), 0.2)
            self.assertLess(rec.structural_failure_rate, 0.05)

    def test_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/perf.json"
            store = ModelPerformanceStore(path=path)
            v_ok = OutputValidationResult(is_valid=True, issues=[], severity="low")
            store.update_model_metrics("persisted", v_ok, 0.9, 1.9, retry_used=False)
            store2 = ModelPerformanceStore(path=path)
            rec = store2.get("persisted")
            assert rec is not None
            self.assertEqual(rec.model_name, "persisted")
            self.assertGreater(rec.total_requests, 0)
