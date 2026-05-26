"""Memory v7.1 — recurring theme aggregation for Memory Manager."""
from __future__ import annotations

from collections import Counter
from typing import Any


def aggregate_recurring_themes(rows: list[dict], *, limit: int = 5) -> list[dict[str, Any]]:
    """Deterministic theme rollup from visible memory rows (no LLM).

    Combines category labels, episode ``topics[]``, and high-frequency query
    fingerprint labels into a sorted count list.
    """
    counts: Counter[str] = Counter()

    for item in rows or []:
        payload = item.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        category = str(payload.get("category") or "").strip().lower()
        if category and category != "context":
            counts[f"category:{category}"] += 1

        for topic in payload.get("topics") or []:
            t = str(topic).strip().lower()
            if t:
                counts[f"topic:{t}"] += 1

        fps = payload.get("retrieval_query_fps") or []
        if isinstance(fps, list):
            for fp in fps:
                label = str(fp).strip().lower()
                if len(label) >= 3:
                    counts[f"query:{label[:48]}"] += 1

    ranked = counts.most_common(max(1, int(limit)))
    return [{"theme": key, "count": count} for key, count in ranked]
