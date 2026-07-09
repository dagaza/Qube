"""Tests for knowledge HTTP client retries (HTTP resilience Slice 4)."""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters import openalex  # noqa: E402
from core.knowledge.http_client import (  # noqa: E402
    BudgetExhaustedError,
    knowledge_get,
    openalex_anonymous_search_throttled,
    openalex_budget_exhausted,
    retry_after_seconds,
    server_error_backoff_sec,
    server_error_wait_sec,
)
from core.knowledge.http_metrics import (  # noqa: E402
    begin_turn_http_metrics,
    reset_http_metrics,
    snapshot_turn_http_summary,
)
from core.knowledge.host_scheduler import get_host_scheduler, reset_host_scheduler  # noqa: E402
from core.knowledge.negative_cache import get_host_negative, reset_negative_cache  # noqa: E402


def _mock_response(*, status: int, headers: dict | None = None, json_body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("not json")
    return resp


class TestRetryHelpers(unittest.TestCase):
    def test_retry_after_seconds_parses_integer(self) -> None:
        self.assertEqual(retry_after_seconds({"Retry-After": "2"}), 2.0)

    def test_retry_after_seconds_parses_http_date(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(seconds=5)
        headers = {"Retry-After": format_datetime(future, usegmt=True)}
        wait = retry_after_seconds(headers, now=datetime.now(timezone.utc))
        self.assertIsNotNone(wait)
        assert wait is not None
        self.assertGreaterEqual(wait, 4.0)
        self.assertLessEqual(wait, 6.0)

    def test_openalex_budget_exhausted(self) -> None:
        self.assertTrue(openalex_budget_exhausted({"X-RateLimit-Remaining": "0"}))
        self.assertFalse(openalex_budget_exhausted({"X-RateLimit-Remaining": "0.5"}))

    def test_openalex_anonymous_search_throttled(self) -> None:
        body = {
            "error": "Search temporarily unavailable",
            "message": "Anonymous search is temporarily rate-limited due to heavy load.",
        }
        resp = _mock_response(status=503, json_body=body)
        self.assertTrue(openalex_anonymous_search_throttled(resp))
        self.assertFalse(openalex_anonymous_search_throttled(_mock_response(status=502)))

    def test_server_error_wait_sec_prefers_retry_after(self) -> None:
        resp = _mock_response(status=503, headers={"Retry-After": "60"})
        self.assertEqual(server_error_wait_sec(resp, 0), 60.0)
        resp_no_header = _mock_response(status=503)
        self.assertGreaterEqual(server_error_wait_sec(resp_no_header, 0), 1.0)

    def test_server_error_backoff_grows_with_attempt(self) -> None:
        first = server_error_backoff_sec(0)
        second = server_error_backoff_sec(2)
        self.assertGreaterEqual(first, 1.0)
        self.assertGreater(second, first)


class TestKnowledgeGetRetries(unittest.TestCase):
    def setUp(self) -> None:
        reset_host_scheduler()
        reset_http_metrics()
        reset_negative_cache()

    def tearDown(self) -> None:
        reset_host_scheduler()
        reset_http_metrics()
        reset_negative_cache()

    @patch("core.knowledge.http_client._sleep")
    @patch("core.knowledge.http_client._execute_once")
    def test_429_honors_retry_after(self, mock_execute, mock_sleep) -> None:
        begin_turn_http_metrics()
        mock_execute.side_effect = [
            _mock_response(status=429, headers={"Retry-After": "1"}),
            _mock_response(status=200, headers={}),
        ]
        resp = knowledge_get("https://api.openalex.org/works")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_execute.call_count, 2)
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 1.0, places=1)

        summary = snapshot_turn_http_summary()
        self.assertIn("429_retry_after", summary["retry_reasons"][0])

    @patch("core.knowledge.http_client._execute_once")
    def test_openalex_budget_exhausted_does_not_retry(self, mock_execute) -> None:
        begin_turn_http_metrics()
        mock_execute.return_value = _mock_response(
            status=429,
            headers={"X-RateLimit-Remaining": "0", "Retry-After": "60"},
        )
        with self.assertRaises(BudgetExhaustedError):
            knowledge_get("https://api.openalex.org/works")
        self.assertEqual(mock_execute.call_count, 1)
        summary = snapshot_turn_http_summary()
        self.assertIn("budget_exhausted", summary["retry_reasons"][0])

    @patch("core.knowledge.http_client._sleep")
    @patch("core.knowledge.http_client._execute_once")
    def test_503_retries_then_returns_last_response(self, mock_execute, mock_sleep) -> None:
        begin_turn_http_metrics()
        mock_execute.side_effect = [_mock_response(status=503)] * 4
        resp = knowledge_get("https://api.openalex.org/works")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(mock_execute.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)
        summary = snapshot_turn_http_summary()
        self.assertEqual(len(summary["retry_reasons"]), 3)
        self.assertTrue(all("503_backoff" in reason for reason in summary["retry_reasons"]))

    @patch("core.knowledge.http_client._sleep")
    @patch("core.knowledge.http_client._execute_once")
    def test_503_honors_retry_after(self, mock_execute, mock_sleep) -> None:
        mock_execute.side_effect = [
            _mock_response(status=503, headers={"Retry-After": "60"}),
            _mock_response(status=200),
        ]
        resp = knowledge_get("https://api.openalex.org/works")
        self.assertEqual(resp.status_code, 200)
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 60.0, places=1)

    @patch("core.knowledge.http_client._execute_once")
    def test_openalex_anonymous_search_throttle_fails_fast(self, mock_execute) -> None:
        begin_turn_http_metrics()
        mock_execute.return_value = _mock_response(
            status=503,
            headers={"Retry-After": "60"},
            json_body={
                "error": "Search temporarily unavailable",
                "message": "Anonymous search is temporarily rate-limited due to heavy load.",
            },
        )
        with self.assertRaises(BudgetExhaustedError):
            knowledge_get("https://api.openalex.org/works")
        self.assertEqual(mock_execute.call_count, 1)
        health = get_host_scheduler().host_health_snapshot()
        self.assertNotEqual(
            health.get("api.openalex.org", {}).get("state"),
            "open",
        )
        entry = get_host_negative("api.openalex.org")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.reason, "budget_exhausted")

    @patch("core.knowledge.http_client._sleep")
    @patch("core.knowledge.http_client._execute_once")
    def test_503_retries_count_one_circuit_failure(self, mock_execute, mock_sleep) -> None:
        mock_execute.side_effect = [_mock_response(status=503)] * 4
        knowledge_get("https://api.openalex.org/works")
        health = get_host_scheduler().host_health_snapshot()
        self.assertNotEqual(
            health.get("api.openalex.org", {}).get("state"),
            "open",
        )
        self.assertEqual(
            health.get("api.openalex.org", {}).get("consecutive_failures", 0),
            1,
        )

    @patch("core.knowledge.adapters.openalex.knowledge_get")
    def test_openalex_adapter_returns_empty_on_budget_exhausted(self, mock_get) -> None:
        mock_get.side_effect = BudgetExhaustedError(
            host="api.openalex.org",
            metrics_host="api.openalex.org",
        )
        rows = openalex.search_openalex("quantum computing", max_results=1)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
