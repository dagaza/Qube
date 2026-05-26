"""
Helpers for merging MemoryUsageRecorder drain deltas into LanceDB JSON payloads.

v7.1: retrieval_days, retrieval_score_sum/count, FIFO caps.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

MAX_RETRIEVAL_DAYS = 16
MAX_RETRIEVAL_QUERY_FPS = 32


def iso_day_from_timestamp(ts: float | None = None) -> str:
    """UTC calendar day bucket ``YYYY-MM-DD``."""
    when = float(ts if ts is not None else time.time())
    return datetime.fromtimestamp(when, tz=timezone.utc).strftime("%Y-%m-%d")


def merge_retrieval_day(existing: list[str] | None, day: str, *, limit: int = MAX_RETRIEVAL_DAYS) -> list[str]:
    days = [str(d) for d in (existing or []) if d]
    if day and day not in days:
        days.append(day)
    return days[-limit:]


def apply_usage_deltas_to_payload(
    payload: dict[str, Any],
    *,
    retrieved: int = 0,
    cited: int = 0,
    query_fps: list[str] | None = None,
    retrieval_scores: list[float] | None = None,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Return a shallow-copied payload with usage counters merged."""
    out = dict(payload)
    ts = float(now_ts if now_ts is not None else time.time())
    out["times_retrieved"] = int(out.get("times_retrieved", 0)) + max(0, int(retrieved))
    out["times_cited_positively"] = int(out.get("times_cited_positively", 0)) + max(0, int(cited))
    out["last_used_at"] = int(ts)

    fps = list(out.get("retrieval_query_fps") or [])
    for fp in query_fps or []:
        if fp and fp not in fps:
            fps.append(str(fp))
    fps = fps[-MAX_RETRIEVAL_QUERY_FPS:]
    out["retrieval_query_fps"] = fps
    out["unique_query_count"] = len(fps)

    if retrieved > 0:
        day = iso_day_from_timestamp(ts)
        out["retrieval_days"] = merge_retrieval_day(out.get("retrieval_days"), day)

    score_sum = float(out.get("retrieval_score_sum") or 0.0)
    score_count = int(out.get("retrieval_score_count") or 0)
    for raw in retrieval_scores or []:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val < 0:
            continue
        score_sum += val
        score_count += 1
    out["retrieval_score_sum"] = score_sum
    out["retrieval_score_count"] = score_count

    return out
