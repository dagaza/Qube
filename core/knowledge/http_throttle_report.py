"""Parse HTTP throttle / resilience signals for eval reporting."""

from __future__ import annotations

import re
from typing import Any, Mapping

_RETRY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("circuit_open", re.compile(r"^([^:]+):circuit_open$")),
    ("budget_exhausted", re.compile(r"^([^:]+):budget_exhausted$")),
    (
        "negative_cache",
        re.compile(r"^([^:]+):negative_cache_(?P<reason>budget_exhausted|circuit_open)$"),
    ),
    (
        "rate_limit_retry",
        re.compile(r"^([^:]+):429_retry_after_(?P<wait>[0-9.]+)s$"),
    ),
    (
        "server_error_backoff",
        re.compile(r"^([^:]+):(?P<code>502|503|504)_backoff_(?P<wait>[0-9.]+)s$"),
    ),
)


def build_throttle_report(http_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Structured throttle view derived from ``http_summary``."""
    if not http_summary:
        return {
            "throttled": False,
            "short_circuit": False,
            "events": [],
            "hosts_open": [],
            "status_429_total": 0,
            "status_503_total": 0,
        }

    events: list[dict[str, Any]] = []
    for reason in http_summary.get("retry_reasons") or []:
        text = str(reason).strip()
        if not text:
            continue
        matched = False
        for kind, pattern in _RETRY_PATTERNS:
            m = pattern.match(text)
            if not m:
                continue
            host = m.group(1)
            event: dict[str, Any] = {"kind": kind, "host": host, "detail": text}
            if kind == "negative_cache":
                event["reason"] = m.group("reason")
            elif kind in {"rate_limit_retry", "server_error_backoff"}:
                event["wait_sec"] = float(m.group("wait"))
                if kind == "server_error_backoff":
                    event["status_code"] = int(m.group("code"))
            events.append(event)
            matched = True
            break
        if not matched:
            events.append({"kind": "other", "host": None, "detail": text})

    hosts_open: list[str] = []
    host_health = http_summary.get("host_health") or {}
    if isinstance(host_health, Mapping):
        for host, row in host_health.items():
            if not isinstance(row, Mapping):
                continue
            if str(row.get("state") or "") == "open":
                hosts_open.append(str(host))

    status_429 = 0
    status_503 = 0
    by_host = http_summary.get("by_host") or {}
    if isinstance(by_host, Mapping):
        for row in by_host.values():
            if not isinstance(row, Mapping):
                continue
            status_429 += int(row.get("429") or 0)
            status_503 += int(row.get("503") or 0)

    short_circuit = any(
        e.get("kind") in {"circuit_open", "budget_exhausted", "negative_cache"}
        for e in events
    ) or bool(hosts_open)

    return {
        "throttled": bool(events or hosts_open or status_429 or status_503),
        "short_circuit": short_circuit,
        "events": events,
        "hosts_open": sorted(hosts_open),
        "status_429_total": status_429,
        "status_503_total": status_503,
    }


def classify_query_failure(status: str, throttle_report: Mapping[str, Any]) -> str | None:
    """Separate retrieval-quality failures from provider throttle failures."""
    if status not in {"no_results", "partial"}:
        return None
    if not throttle_report.get("throttled"):
        return "retrieval"
    if throttle_report.get("short_circuit"):
        return "throttle"
    if status == "no_results" and (
        int(throttle_report.get("status_429_total") or 0) > 0
        or int(throttle_report.get("status_503_total") or 0) > 0
    ):
        return "mixed"
    if status == "no_results":
        return "retrieval"
    return "retrieval"


def aggregate_throttle_reports(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Corpus-level throttle summary for eval JSON output."""
    throttled_queries = sum(1 for r in reports if r.get("throttled"))
    short_circuit_queries = sum(1 for r in reports if r.get("short_circuit"))
    failure_classes: dict[str, int] = {}
    hosts_open: set[str] = set()
    status_429 = 0
    status_503 = 0
    events: list[dict[str, Any]] = []

    for report in reports:
        if not report.get("throttled"):
            continue
        status_429 += int(report.get("status_429_total") or 0)
        status_503 += int(report.get("status_503_total") or 0)
        for host in report.get("hosts_open") or []:
            hosts_open.add(str(host))
        for event in report.get("events") or []:
            if isinstance(event, Mapping):
                events.append(dict(event))

    return {
        "queries_throttled": throttled_queries,
        "queries_short_circuited": short_circuit_queries,
        "hosts_open": sorted(hosts_open),
        "status_429_total": status_429,
        "status_503_total": status_503,
        "events": events,
    }


def attach_throttle_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Add ``throttle_report`` and optional ``failure_class`` to an eval row."""
    http_summary = payload.get("http_summary")
    report = build_throttle_report(
        http_summary if isinstance(http_summary, Mapping) else None
    )
    payload["throttle_report"] = report
    failure_class = classify_query_failure(str(payload.get("status") or ""), report)
    if failure_class is not None:
        payload["failure_class"] = failure_class
    return payload
