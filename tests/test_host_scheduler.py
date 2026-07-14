"""Tests for knowledge host scheduler (HTTP resilience Slice 3)."""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.host_scheduler import (  # noqa: E402
    HostScheduler,
    SerializedInterval,
    TokenBucket,
    metrics_host_for,
    reset_host_scheduler,
    scheduler_key_for_host,
)
from core.knowledge.http_client import knowledge_get  # noqa: E402
from core.knowledge.http_metrics import (  # noqa: E402
    begin_turn_http_metrics,
    reset_http_metrics,
    snapshot_turn_http_summary,
)


class TestSchedulerKeys(unittest.TestCase):
    def test_ncbi_hosts_share_bucket(self) -> None:
        self.assertEqual(
            scheduler_key_for_host("eutils.ncbi.nlm.nih.gov"),
            "ncbi",
        )
        self.assertEqual(
            scheduler_key_for_host("pubchem.ncbi.nlm.nih.gov"),
            "ncbi",
        )
        self.assertEqual(metrics_host_for("pubchem.ncbi.nlm.nih.gov"), "ncbi")

    def test_other_hosts_use_hostname(self) -> None:
        self.assertEqual(scheduler_key_for_host("api.openalex.org"), "api.openalex.org")


class TestTokenBucket(unittest.TestCase):
    def test_enforces_rate_between_requests(self) -> None:
        bucket = TokenBucket(rate_per_sec=10.0, burst=1.0)
        t0 = time.monotonic()
        bucket.acquire()
        bucket.acquire()
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.08)


class TestSerializedInterval(unittest.TestCase):
    def test_parallel_requests_serialize(self) -> None:
        gate = SerializedInterval(min_interval_sec=0.08)
        times: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            gate.acquire()
            with lock:
                times.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        gap = sorted(times)[1] - sorted(times)[0]
        self.assertGreaterEqual(gap, 0.07)


class TestHostSchedulerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        reset_host_scheduler()
        reset_http_metrics()

    def tearDown(self) -> None:
        reset_host_scheduler()
        reset_http_metrics()

    def test_ncbi_shared_bucket_limits_cross_host_burst(self) -> None:
        scheduler = HostScheduler()
        stamps: list[float] = []
        lock = threading.Lock()

        def hit(host: str) -> None:
            scheduler.acquire(host)
            with lock:
                stamps.append(time.monotonic())

        with patch("core.knowledge.host_scheduler.ncbi_rate_per_sec", return_value=10.0):
            with patch(
                "core.knowledge.host_scheduler._HOST_POLICIES",
                {
                    "ncbi": __import__(
                        "core.knowledge.host_scheduler", fromlist=["HostPolicySpec"]
                    ).HostPolicySpec(
                        kind="token_bucket",
                        rate_per_sec=None,
                        burst=1.0,
                    )
                },
            ):
                reset_host_scheduler()
                scheduler = HostScheduler()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    pool.submit(hit, "eutils.ncbi.nlm.nih.gov")
                    pool.submit(hit, "pubchem.ncbi.nlm.nih.gov")
                    pool.submit(hit, "eutils.ncbi.nlm.nih.gov")
        ordered = sorted(stamps)
        self.assertEqual(len(ordered), 3)
        self.assertGreaterEqual(ordered[1] - ordered[0], 0.08)
        self.assertGreaterEqual(ordered[2] - ordered[1], 0.08)

    def test_cross_host_parallel_not_blocked_by_single_arxiv_slot(self) -> None:
        scheduler = HostScheduler()
        events: list[tuple[str, float]] = []
        lock = threading.Lock()

        def hit(host: str) -> None:
            scheduler.acquire(host)
            with lock:
                events.append((host, time.monotonic()))

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=3) as pool:
            pool.submit(hit, "export.arxiv.org")
            pool.submit(hit, "api.openalex.org")
            pool.submit(hit, "inspirehep.net")
        elapsed = time.monotonic() - t0
        hosts = {host for host, _ in events}
        self.assertEqual(hosts, {"export.arxiv.org", "api.openalex.org", "inspirehep.net"})
        self.assertLess(elapsed, 1.0)

    def test_arxiv_serializes_back_to_back_requests(self) -> None:
        scheduler = HostScheduler()
        t0 = time.monotonic()
        scheduler.acquire("export.arxiv.org")
        scheduler.acquire("export.arxiv.org")
        self.assertGreaterEqual(time.monotonic() - t0, 3.4)

    @patch("core.knowledge.http_client.requests.get")
    def test_chemistry_adapters_share_ncbi_metrics_bucket(self, mock_get: MagicMock) -> None:
        from core.knowledge.adapters import pubchem, pubmed_eutils

        begin_turn_http_metrics()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = resp

        pubmed_eutils.search_pubmed("aspirin binding", max_results=1)
        resp.json.return_value = {"IdentifierList": {"CID": []}}
        pubchem._fetch_cid("aspirin", timeout=1.0)

        summary = snapshot_turn_http_summary()
        self.assertIn("ncbi", summary["by_host"])
        self.assertGreaterEqual(summary["by_host"]["ncbi"]["requests"], 2)


class TestKnowledgeGetScheduling(unittest.TestCase):
    def setUp(self) -> None:
        reset_host_scheduler()
        reset_http_metrics()

    @patch("core.knowledge.http_client.requests.get")
    def test_knowledge_get_acquires_scheduler_slot(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        mock_get.return_value = resp
        scheduler = HostScheduler()
        with patch("core.knowledge.http_client.get_host_scheduler", return_value=scheduler):
            with patch.object(scheduler, "acquire") as acquire:
                knowledge_get("https://api.openalex.org/works")
        acquire.assert_called_once_with("api.openalex.org")


if __name__ == "__main__":
    unittest.main()
