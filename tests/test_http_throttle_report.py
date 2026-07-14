"""Tests for HTTP throttle eval reporting (Slice 7 partial)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.http_throttle_report import (  # noqa: E402
    aggregate_throttle_reports,
    attach_throttle_fields,
    build_throttle_report,
    classify_query_failure,
)


class TestHttpThrottleReport(unittest.TestCase):
    def test_build_throttle_report_parses_retry_reasons(self) -> None:
        report = build_throttle_report(
            {
                "retry_reasons": [
                    "api.openalex.org:429_retry_after_1.0s",
                    "api.openalex.org:circuit_open",
                    "ncbi:negative_cache_budget_exhausted",
                ],
                "host_health": {
                    "api.openalex.org": {"state": "open", "consecutive_failures": 3},
                },
                "by_host": {
                    "api.openalex.org": {"requests": 2, "429": 1, "503": 0, "retries": 1},
                },
            }
        )
        self.assertTrue(report["throttled"])
        self.assertTrue(report["short_circuit"])
        self.assertEqual(report["status_429_total"], 1)
        self.assertEqual(report["hosts_open"], ["api.openalex.org"])
        kinds = {e["kind"] for e in report["events"]}
        self.assertIn("rate_limit_retry", kinds)
        self.assertIn("circuit_open", kinds)
        self.assertIn("negative_cache", kinds)

    def test_classify_throttle_vs_retrieval(self) -> None:
        throttle = build_throttle_report(
            {"retry_reasons": ["api.openalex.org:negative_cache_circuit_open"]}
        )
        self.assertEqual(classify_query_failure("no_results", throttle), "throttle")
        clean = build_throttle_report({"by_host": {}})
        self.assertEqual(classify_query_failure("no_results", clean), "retrieval")
        mixed = build_throttle_report(
            {
                "retry_reasons": ["api.openalex.org:429_retry_after_2.0s"],
                "by_host": {"api.openalex.org": {"requests": 2, "429": 1, "503": 0}},
            }
        )
        self.assertEqual(classify_query_failure("no_results", mixed), "mixed")

    def test_attach_throttle_fields(self) -> None:
        row = attach_throttle_fields(
            {
                "id": "soc_001",
                "status": "no_results",
                "http_summary": {
                    "retry_reasons": ["api.openalex.org:budget_exhausted"],
                },
            }
        )
        self.assertIn("throttle_report", row)
        self.assertEqual(row["failure_class"], "throttle")

    def test_aggregate_throttle_reports(self) -> None:
        agg = aggregate_throttle_reports(
            [
                {"throttled": True, "short_circuit": True, "hosts_open": ["api.openalex.org"], "events": [], "status_429_total": 0, "status_503_total": 0},
                {"throttled": False, "short_circuit": False, "hosts_open": [], "events": [], "status_429_total": 0, "status_503_total": 0},
            ]
        )
        self.assertEqual(agg["queries_throttled"], 1)
        self.assertEqual(agg["queries_short_circuited"], 1)


if __name__ == "__main__":
    unittest.main()
