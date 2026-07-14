"""Tests for knowledge HTTP metrics (Slice 1)."""

from __future__ import annotations

import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.http_metrics import (  # noqa: E402
    HttpMetricsCollector,
    begin_turn_http_metrics,
    build_http_summary,
    format_http_report,
    hostname_from_url,
    instrumented_get,
    merge_http_summaries,
    record_http_request,
    reset_http_metrics,
    snapshot_turn_http_summary,
)
from core.knowledge.host_scheduler import reset_host_scheduler  # noqa: E402
from core.knowledge.negative_cache import reset_negative_cache  # noqa: E402
from core.knowledge.pipeline_scientific import ScientificEvidencePipeline  # noqa: E402
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402


class TestHttpMetrics(unittest.TestCase):
    def setUp(self) -> None:
        reset_http_metrics()
        reset_negative_cache()
        reset_host_scheduler()

    def tearDown(self) -> None:
        reset_http_metrics()
        reset_negative_cache()
        reset_host_scheduler()

    def test_hostname_from_url(self) -> None:
        self.assertEqual(hostname_from_url("https://api.openalex.org/works"), "api.openalex.org")

    def test_collector_records_status_and_latency(self) -> None:
        collector = HttpMetricsCollector()
        collector.record(
            host="api.openalex.org",
            status_code=429,
            latency_ms=120.0,
            is_retry=False,
            headers={"X-RateLimit-Remaining": "0.94"},
        )
        collector.record(
            host="api.openalex.org",
            status_code=200,
            latency_ms=80.0,
            is_retry=True,
        )
        summary = collector.snapshot()
        self.assertEqual(summary["requests_total"], 2)
        host = summary["by_host"]["api.openalex.org"]
        self.assertEqual(host["429"], 1)
        self.assertEqual(host["retries"], 1)
        self.assertEqual(host["rate_limit_remaining"], 0.94)
        self.assertIn("latency_ms_p95", host)

    def test_turn_scope_snapshot_clears_active_turn(self) -> None:
        begin_turn_http_metrics()
        record_http_request(host="export.arxiv.org", status_code=200, latency_ms=50.0)
        summary = snapshot_turn_http_summary()
        self.assertEqual(summary["requests_total"], 1)
        self.assertEqual(snapshot_turn_http_summary()["requests_total"], 0)

    def test_merge_http_summaries(self) -> None:
        merged = merge_http_summaries(
            [
                {
                    "requests_total": 2,
                    "cache_hits_evidence": 0,
                    "by_host": {
                        "api.openalex.org": {"requests": 2, "429": 1, "503": 0, "retries": 1},
                    },
                },
                {
                    "requests_total": 3,
                    "cache_hits_evidence": 1,
                    "by_host": {
                        "api.openalex.org": {"requests": 1, "429": 0, "503": 1, "retries": 0},
                        "eutils.ncbi.nlm.nih.gov": {"requests": 2, "429": 0, "503": 0, "retries": 0},
                    },
                },
            ]
        )
        self.assertEqual(merged["requests_total"], 5)
        self.assertEqual(merged["cache_hits_evidence"], 1)
        openalex = merged["by_host"]["api.openalex.org"]
        self.assertEqual(openalex["requests"], 3)
        self.assertEqual(openalex["429"], 1)
        self.assertEqual(openalex["503"], 1)

    def test_format_http_report(self) -> None:
        text = format_http_report(
            {
                "requests_total": 5,
                "cache_hits_evidence": 1,
                "by_host": {
                    "api.openalex.org": {"requests": 3, "429": 1, "503": 0, "retries": 1},
                },
            }
        )
        self.assertIn("5 requests", text)
        self.assertIn("1×429", text)
        self.assertIn("api.openalex.org", text)

    @patch("core.knowledge.http_client.requests.get")
    def test_instrumented_get_records_response(self, mock_get: MagicMock) -> None:
        begin_turn_http_metrics()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        mock_get.return_value = resp
        out = instrumented_get("https://api.openalex.org/works", timeout=1.0)
        self.assertIs(out, resp)
        summary = snapshot_turn_http_summary()
        host = summary["by_host"]["api.openalex.org"]
        self.assertEqual(host["requests"], 1)

    def test_thread_safe_recording(self) -> None:
        begin_turn_http_metrics()

        def worker() -> None:
            for _ in range(20):
                record_http_request(host="api.openalex.org", status_code=200, latency_ms=10.0)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        summary = snapshot_turn_http_summary()
        self.assertEqual(summary["by_host"]["api.openalex.org"]["requests"], 80)

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    @patch("core.knowledge.pipeline_scientific.resolve_service_adapters", return_value=["openalex"])
    @patch("core.knowledge.pipeline_scientific.get_search_function")
    def test_pipeline_attaches_http_summary(
        self,
        mock_get_fn,
        _mock_resolve,
        _mock_set_cache,
        _mock_get_cache,
    ) -> None:
        def _search(_query: str, *, max_results: int = 3):
            record_http_request(host="api.openalex.org", status_code=200, latency_ms=42.0)
            return [
                {
                    "title": "Sample work",
                    "snippet": "Sample abstract about quantum computing research.",
                    "full_text": "Sample abstract about quantum computing research.",
                    "url": "https://openalex.org/W1",
                    "_adapter": "openalex",
                }
            ]

        mock_get_fn.return_value = _search
        ctx = RetrievalContext(
            query="quantum computing",
            semantic_query="quantum computing",
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            budget=RetrievalBudget(max_results=3),
        )
        _bundle, rel_diag, _raw = ScientificEvidencePipeline().run(ctx)
        self.assertIsNotNone(rel_diag)
        assert rel_diag is not None
        summary = rel_diag.get("http_summary")
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["requests_total"], 1)
        self.assertIn("api.openalex.org", summary["by_host"])

    @patch("core.knowledge.pipeline_scientific.get_cached_rows")
    @patch("core.knowledge.pipeline_scientific.resolve_service_adapters", return_value=["openalex"])
    def test_pipeline_cache_hit_reports_zero_http(
        self,
        _mock_resolve,
        mock_get_cache,
    ) -> None:
        mock_get_cache.return_value = [
            {
                "title": "Cached work",
                "snippet": "Cached abstract text.",
                "full_text": "Cached abstract text.",
                "url": "https://openalex.org/W2",
                "_adapter": "openalex",
            }
        ]
        ctx = RetrievalContext(
            query="quantum computing",
            semantic_query="quantum computing",
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            budget=RetrievalBudget(max_results=3),
        )
        _bundle, rel_diag, _raw = ScientificEvidencePipeline().run(ctx)
        self.assertIsNotNone(rel_diag)
        assert rel_diag is not None
        summary = rel_diag["http_summary"]
        self.assertEqual(summary["requests_total"], 0)
        self.assertEqual(summary["cache_hits_evidence"], 1)


if __name__ == "__main__":
    unittest.main()
