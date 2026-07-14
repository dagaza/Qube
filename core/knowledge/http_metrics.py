"""Thread-safe HTTP observability for knowledge adapter outbound calls (Slice 1)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

from core.knowledge.host_scheduler import host_health_snapshot

_MAX_LATENCY_SAMPLES = 256


def hostname_from_url(url: str) -> str:
    """Return normalized hostname for metrics grouping."""
    host = (urlparse(url).hostname or "").strip().lower()
    return host or "unknown"


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    value = ordered[low] * (1.0 - weight) + ordered[high] * weight
    return round(value, 1)


@dataclass
class HostHttpStats:
    requests: int = 0
    status_429: int = 0
    status_503: int = 0
    retries: int = 0
    latency_ms_samples: list[float] = field(default_factory=list)
    rate_limit_remaining: float | None = None
    last_request_at: float = 0.0

    def record(
        self,
        *,
        status_code: int,
        latency_ms: float,
        is_retry: bool,
        headers: Mapping[str, str] | None,
    ) -> None:
        self.requests += 1
        self.last_request_at = time.time()
        if is_retry:
            self.retries += 1
        if status_code == 429:
            self.status_429 += 1
        elif status_code == 503:
            self.status_503 += 1
        self.latency_ms_samples.append(latency_ms)
        if len(self.latency_ms_samples) > _MAX_LATENCY_SAMPLES:
            self.latency_ms_samples = self.latency_ms_samples[-_MAX_LATENCY_SAMPLES:]
        if headers:
            remaining = headers.get("X-RateLimit-Remaining")
            if remaining is not None:
                try:
                    self.rate_limit_remaining = float(remaining)
                except (TypeError, ValueError):
                    pass


class HttpMetricsCollector:
    """Process-wide, thread-safe HTTP metrics accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_host: dict[str, HostHttpStats] = {}
        self._retry_reasons: list[str] = []

    def reset(self) -> None:
        with self._lock:
            self._by_host.clear()
            self._retry_reasons.clear()

    def record(
        self,
        *,
        host: str,
        status_code: int,
        latency_ms: float,
        is_retry: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        key = (host or "unknown").strip().lower() or "unknown"
        with self._lock:
            stats = self._by_host.setdefault(key, HostHttpStats())
            stats.record(
                status_code=status_code,
                latency_ms=latency_ms,
                is_retry=is_retry,
                headers=headers,
            )

    def record_retry_reason(self, reason: str) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        with self._lock:
            self._retry_reasons.append(text)

    def snapshot(self, *, cache_hits_evidence: int = 0) -> dict[str, Any]:
        with self._lock:
            by_host = {host: stats for host, stats in self._by_host.items()}
            retry_reasons = list(self._retry_reasons)
        summary = build_http_summary(by_host, cache_hits_evidence=cache_hits_evidence)
        if retry_reasons:
            summary["retry_reasons"] = retry_reasons
        health = host_health_snapshot()
        if health:
            summary["host_health"] = health
        return summary


_collector = HttpMetricsCollector()
_turn_collector: HttpMetricsCollector | None = None
_turn_lock = threading.Lock()


def reset_http_metrics() -> None:
    """Clear the global metrics accumulator (eval harness entry point)."""
    _collector.reset()


def global_http_summary(*, cache_hits_evidence: int = 0) -> dict[str, Any]:
    """Process-wide HTTP summary for provider status aggregation."""
    return _collector.snapshot(cache_hits_evidence=cache_hits_evidence)


def begin_turn_http_metrics() -> None:
    """Start a fresh per-turn metrics scope (scientific pipeline entry)."""
    global _turn_collector
    turn = HttpMetricsCollector()
    with _turn_lock:
        _turn_collector = turn


def snapshot_turn_http_summary(*, cache_hits_evidence: int = 0) -> dict[str, Any]:
    """Return HTTP summary for the active turn and clear the turn scope."""
    global _turn_collector
    with _turn_lock:
        turn = _turn_collector
        _turn_collector = None
    if turn is not None:
        return turn.snapshot(cache_hits_evidence=cache_hits_evidence)
    return _collector.snapshot(cache_hits_evidence=cache_hits_evidence)


def _active_collector() -> HttpMetricsCollector:
    with _turn_lock:
        if _turn_collector is not None:
            return _turn_collector
    return _collector


def record_http_request(
    *,
    host: str,
    status_code: int,
    latency_ms: float,
    is_retry: bool = False,
    headers: Mapping[str, str] | None = None,
) -> None:
    _active_collector().record(
        host=host,
        status_code=status_code,
        latency_ms=latency_ms,
        is_retry=is_retry,
        headers=headers,
    )


def record_http_retry_reason(reason: str) -> None:
    _active_collector().record_retry_reason(reason)


def build_http_summary(
    by_host: Mapping[str, HostHttpStats],
    *,
    cache_hits_evidence: int = 0,
) -> dict[str, Any]:
    host_payload: dict[str, Any] = {}
    requests_total = 0
    for host, stats in sorted(by_host.items()):
        requests_total += stats.requests
        entry: dict[str, Any] = {
            "requests": stats.requests,
            "429": stats.status_429,
            "503": stats.status_503,
            "retries": stats.retries,
        }
        p95 = _percentile(stats.latency_ms_samples, 0.95)
        if p95 is not None:
            entry["latency_ms_p95"] = p95
        if stats.rate_limit_remaining is not None:
            entry["rate_limit_remaining"] = stats.rate_limit_remaining
        if stats.last_request_at > 0:
            entry["last_request_at"] = stats.last_request_at
        host_payload[host] = entry
    return {
        "requests_total": requests_total,
        "cache_hits_evidence": cache_hits_evidence,
        "by_host": host_payload,
    }


def merge_http_summaries(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge per-query HTTP summaries (eval harness aggregation)."""
    merged_hosts: dict[str, HostHttpStats] = {}
    cache_hits = 0
    for summary in summaries:
        if not summary:
            continue
        try:
            cache_hits += int(summary.get("cache_hits_evidence") or 0)
        except (TypeError, ValueError):
            pass
        by_host = summary.get("by_host") or {}
        if not isinstance(by_host, Mapping):
            continue
        for host, row in by_host.items():
            if not isinstance(row, Mapping):
                continue
            stats = merged_hosts.setdefault(str(host), HostHttpStats())
            try:
                stats.requests += int(row.get("requests") or 0)
                stats.status_429 += int(row.get("429") or 0)
                stats.status_503 += int(row.get("503") or 0)
                stats.retries += int(row.get("retries") or 0)
            except (TypeError, ValueError):
                pass
            p95 = row.get("latency_ms_p95")
            if p95 is not None:
                try:
                    stats.latency_ms_samples.append(float(p95))
                except (TypeError, ValueError):
                    pass
            remaining = row.get("rate_limit_remaining")
            if remaining is not None:
                try:
                    stats.rate_limit_remaining = float(remaining)
                except (TypeError, ValueError):
                    pass
    merged = build_http_summary(merged_hosts, cache_hits_evidence=cache_hits)
    retry_reasons: list[str] = []
    for summary in summaries:
        if not summary:
            continue
        reasons = summary.get("retry_reasons") or []
        if isinstance(reasons, list):
            retry_reasons.extend(str(r) for r in reasons if str(r).strip())
    if retry_reasons:
        merged["retry_reasons"] = retry_reasons
    return merged


def format_http_report(summary: Mapping[str, Any]) -> str:
    """Human-readable HTTP report for eval stderr."""
    total = int(summary.get("requests_total") or 0)
    cache_hits = int(summary.get("cache_hits_evidence") or 0)
    by_host = summary.get("by_host") or {}
    status_429 = 0
    status_503 = 0
    if isinstance(by_host, Mapping):
        for row in by_host.values():
            if not isinstance(row, Mapping):
                continue
            status_429 += int(row.get("429") or 0)
            status_503 += int(row.get("503") or 0)
    lines = [
        f"HTTP: {total} requests ({status_429}×429, {status_503}×503), "
        f"{cache_hits} evidence cache hits",
    ]
    if isinstance(by_host, Mapping):
        for host, row in sorted(by_host.items()):
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"  {host}: {int(row.get('requests') or 0)} req, "
                f"429={int(row.get('429') or 0)}, "
                f"503={int(row.get('503') or 0)}, "
                f"retries={int(row.get('retries') or 0)}"
            )
    return "\n".join(lines)


def instrumented_get(
    url: str,
    *,
    host: str | None = None,
    is_retry: bool = False,
    **kwargs: Any,
) -> Any:
    """Rate-limited knowledge GET (delegates to ``http_client.knowledge_get``)."""
    from core.knowledge.http_client import knowledge_get

    return knowledge_get(url, host=host, is_retry=is_retry, **kwargs)
