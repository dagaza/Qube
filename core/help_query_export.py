"""Aggregate @help query telemetry from Qube.Help log lines (§13)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HELP_LOG_MARKER = "[Help] "
_QUERY_NORM = re.compile(r"[^a-z0-9]+")


def normalize_help_query(text: str) -> str:
    return _QUERY_NORM.sub(" ", (text or "").casefold()).strip()


def parse_help_log_line(line: str) -> dict[str, Any] | None:
    """Parse one log line emitted by ``log_help_query``."""
    if HELP_LOG_MARKER not in line:
        return None
    payload_raw = line.split(HELP_LOG_MARKER, 1)[1].strip()
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("event") != "help_query":
        return None
    return payload


def iter_help_log_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        parsed = parse_help_log_line(line)
        if parsed is not None:
            events.append(parsed)
    return events


def load_help_log_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return iter_help_log_events(text.splitlines())


@dataclass(frozen=True)
class HelpQueryAggregate:
    query: str
    count: int
    empty_retrieval_count: int
    canonical_hits: int
    top_doc_ids: list[tuple[str, int]]


def aggregate_help_queries(events: list[dict[str, Any]]) -> list[HelpQueryAggregate]:
    """Roll up help_query events by normalized query text."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    display: dict[str, str] = {}
    for event in events:
        raw = str(event.get("query") or "").strip()
        key = normalize_help_query(raw) or raw.casefold()
        grouped[key].append(event)
        if key not in display or len(raw) > len(display[key]):
            display[key] = raw

    rows: list[HelpQueryAggregate] = []
    for key, bucket in grouped.items():
        doc_counter: Counter[str] = Counter()
        empty = 0
        canonical_hits = 0
        for event in bucket:
            doc_ids = event.get("retrieved_doc_ids") or []
            if not doc_ids:
                empty += 1
            for doc_id in doc_ids:
                doc_counter[str(doc_id)] += 1
            if event.get("canonical_id"):
                canonical_hits += 1
        rows.append(
            HelpQueryAggregate(
                query=display.get(key, key),
                count=len(bucket),
                empty_retrieval_count=empty,
                canonical_hits=canonical_hits,
                top_doc_ids=doc_counter.most_common(5),
            )
        )

    rows.sort(key=lambda row: (-row.count, row.query.casefold()))
    return rows


def rank_doc_backlog(
    aggregates: list[HelpQueryAggregate],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Priority list for doc work using §13.3:
    (frequency) × (1 − retrieval success) × frustration proxy.

    Frustration proxy (v1): empty retrieval on a repeated query.
    """
    backlog: list[dict[str, Any]] = []
    for row in aggregates:
        if row.count <= 0:
            continue
        success_rate = 1.0 - (row.empty_retrieval_count / row.count)
        frustration = min(1.0, row.empty_retrieval_count / max(1, row.count))
        score = row.count * (1.0 - success_rate) * (0.5 + frustration)
        if score <= 0:
            continue
        backlog.append(
            {
                "query": row.query,
                "count": row.count,
                "empty_retrieval_count": row.empty_retrieval_count,
                "priority_score": round(score, 3),
                "suggested_action": (
                    "canonical answer"
                    if row.empty_retrieval_count >= row.count // 2
                    else "FAQ or troubleshooting refresh"
                ),
            }
        )
    backlog.sort(key=lambda item: (-item["priority_score"], -item["count"]))
    return backlog[:limit]


def export_help_query_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates = aggregate_help_queries(events)
    return {
        "total_events": len(events),
        "unique_queries": len(aggregates),
        "top_queries": [
            {
                "query": row.query,
                "count": row.count,
                "empty_retrieval_count": row.empty_retrieval_count,
                "canonical_hits": row.canonical_hits,
                "top_doc_ids": row.top_doc_ids,
            }
            for row in aggregates[:20]
        ],
        "doc_backlog": rank_doc_backlog(aggregates),
    }
