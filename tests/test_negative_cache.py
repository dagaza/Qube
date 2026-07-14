"""Tests for host negative cache (HTTP resilience Slice 5)."""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.http_client import BudgetExhaustedError, knowledge_get  # noqa: E402
from core.knowledge.http_metrics import begin_turn_http_metrics, snapshot_turn_http_summary  # noqa: E402
from core.knowledge.host_scheduler import reset_host_scheduler  # noqa: E402
from core.knowledge.negative_cache import (  # noqa: E402
    get_host_negative,
    mark_host_negative,
    reset_negative_cache,
)


class TestNegativeCache(unittest.TestCase):
    def setUp(self) -> None:
        reset_negative_cache()
        reset_host_scheduler()

    def tearDown(self) -> None:
        reset_negative_cache()
        reset_host_scheduler()
        os.environ.pop("QUBE_NEGATIVE_CACHE", None)
        os.environ.pop("QUBE_NEGATIVE_CACHE_TTL", None)

    def test_mark_and_lookup(self) -> None:
        mark_host_negative("api.openalex.org", reason="budget_exhausted", ttl_seconds=60)
        entry = get_host_negative("api.openalex.org")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.reason, "budget_exhausted")

    def test_expired_entry_cleared(self) -> None:
        with patch("core.knowledge.negative_cache.time.time", return_value=1000.0):
            mark_host_negative("ncbi", reason="budget_exhausted", ttl_seconds=1)
        with patch("core.knowledge.negative_cache.time.time", return_value=1002.0):
            self.assertIsNone(get_host_negative("ncbi"))

    @patch("core.knowledge.http_client._execute_once")
    def test_knowledge_get_short_circuits_on_negative_cache(self, mock_execute) -> None:
        begin_turn_http_metrics()
        mark_host_negative("api.openalex.org", reason="budget_exhausted", ttl_seconds=300)
        with self.assertRaises(BudgetExhaustedError):
            knowledge_get("https://api.openalex.org/works")
        mock_execute.assert_not_called()
        summary = snapshot_turn_http_summary()
        self.assertIn("negative_cache_budget_exhausted", summary["retry_reasons"][0])


if __name__ == "__main__":
    unittest.main()
