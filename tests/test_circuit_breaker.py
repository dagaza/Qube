"""Tests for host circuit breaker (HTTP resilience Slice 6)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.host_scheduler import (  # noqa: E402
    circuit_open_cooldown_sec,
    get_host_scheduler,
    reset_host_scheduler,
)
from core.knowledge.http_client import HostUnavailableError, knowledge_get  # noqa: E402
from core.knowledge.http_metrics import begin_turn_http_metrics, snapshot_turn_http_summary  # noqa: E402
from core.knowledge.negative_cache import get_host_negative, reset_negative_cache  # noqa: E402


def _mock_response(*, status: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {}
    return resp


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self) -> None:
        reset_host_scheduler()
        reset_negative_cache()
        os.environ["QUBE_CIRCUIT_FAILURE_THRESHOLD"] = "3"
        os.environ["QUBE_CIRCUIT_FAILURE_WINDOW_SEC"] = "60"
        os.environ["QUBE_CIRCUIT_OPEN_COOLDOWN_SEC"] = "300"

    def tearDown(self) -> None:
        reset_host_scheduler()
        reset_negative_cache()
        for name in (
            "QUBE_CIRCUIT_BREAKER",
            "QUBE_CIRCUIT_FAILURE_THRESHOLD",
            "QUBE_CIRCUIT_FAILURE_WINDOW_SEC",
            "QUBE_CIRCUIT_OPEN_COOLDOWN_SEC",
        ):
            os.environ.pop(name, None)

    def test_opens_after_three_failures(self) -> None:
        scheduler = get_host_scheduler()
        for _ in range(3):
            scheduler.record_outcome("api.openalex.org", 503)
        health = scheduler.host_health_snapshot()
        self.assertEqual(health["api.openalex.org"]["state"], "open")
        self.assertIsNotNone(get_host_negative("api.openalex.org"))

    @patch("core.knowledge.http_client._execute_once")
    def test_open_circuit_short_circuits_without_http(self, mock_execute) -> None:
        scheduler = get_host_scheduler()
        for _ in range(3):
            scheduler.record_outcome("api.openalex.org", 503)
        mock_execute.reset_mock()
        with self.assertRaises(HostUnavailableError) as ctx:
            knowledge_get("https://api.openalex.org/works")
        self.assertEqual(ctx.exception.reason, "circuit_open")
        mock_execute.assert_not_called()

    @patch("core.knowledge.http_client.requests.get")
    def test_half_open_probe_success_closes_circuit(self, mock_get) -> None:
        scheduler = get_host_scheduler()
        opened_at = 1000.0
        with patch("core.knowledge.host_scheduler.time.monotonic", return_value=opened_at):
            for _ in range(3):
                scheduler.record_outcome("api.openalex.org", 503)
        circuit = scheduler._get_circuit("api.openalex.org")
        circuit.opened_at = opened_at

        mock_get.return_value = _mock_response(status=200)
        cooldown = circuit_open_cooldown_sec()
        with patch(
            "core.knowledge.host_scheduler.time.monotonic",
            return_value=opened_at + cooldown + 1.0,
        ):
            resp = knowledge_get("https://api.openalex.org/works")
        self.assertEqual(resp.status_code, 200)
        health = scheduler.host_health_snapshot()
        self.assertEqual(health["api.openalex.org"]["state"], "closed")
        self.assertIsNone(get_host_negative("api.openalex.org"))

    @patch("core.knowledge.http_client._execute_once")
    def test_other_hosts_unaffected_when_one_circuit_open(self, mock_execute) -> None:
        scheduler = get_host_scheduler()
        for _ in range(3):
            scheduler.record_outcome("api.openalex.org", 503)
        mock_execute.return_value = _mock_response(status=200)

        def route(url: str, **kwargs):
            if "openalex" in url:
                raise HostUnavailableError(
                    host="api.openalex.org",
                    metrics_host="api.openalex.org",
                    reason="circuit_open",
                )
            return _mock_response(status=200)

        mock_execute.side_effect = route
        with self.assertRaises(HostUnavailableError):
            knowledge_get("https://api.openalex.org/works")
        resp = knowledge_get("https://inspirehep.net/api/literature")
        self.assertEqual(resp.status_code, 200)

    @patch("core.knowledge.http_client._execute_once")
    def test_host_health_in_http_summary(self, mock_execute) -> None:
        begin_turn_http_metrics()
        get_host_scheduler().record_outcome("api.openalex.org", 503)
        get_host_scheduler().record_outcome("api.openalex.org", 503)
        summary = snapshot_turn_http_summary()
        self.assertIn("host_health", summary)
        self.assertIn("api.openalex.org", summary["host_health"])

    @patch("core.knowledge.http_client._sleep")
    @patch("core.knowledge.http_client._execute_once")
    def test_three_failed_knowledge_get_calls_open_circuit(
        self, mock_execute, mock_sleep
    ) -> None:
        mock_execute.side_effect = [_mock_response(status=503)] * 12
        for _ in range(3):
            knowledge_get("https://api.openalex.org/works")
        health = get_host_scheduler().host_health_snapshot()
        self.assertEqual(health["api.openalex.org"]["state"], "open")
        mock_execute.reset_mock()
        with self.assertRaises(HostUnavailableError):
            knowledge_get("https://api.openalex.org/works")
        mock_execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
