"""Ranking profile templates for generic orchestration."""

from __future__ import annotations

from typing import Any


def score_row(row: dict[str, Any], *, query: str, profile: str) -> float:
    title = str(row.get("title") or "").lower()
    snippet = str(row.get("snippet") or "").lower()
    q = (query or "").lower()
    tokens = [t for t in q.split() if len(t) > 2]
    if not tokens:
        return 0.35

    overlap = sum(1 for t in tokens if t in title or t in snippet)
    base = overlap / max(1, len(tokens))

    profile_id = (profile or "generic").strip().lower()
    bonus = 0.0
    if profile_id == "literature":
        if row.get("doi") or row.get("peer_reviewed"):
            bonus += 0.15
        if row.get("preprint"):
            bonus += 0.05
    elif profile_id == "regulatory":
        venue = str(row.get("venue") or "").lower()
        if any(x in venue for x in ("sec", "court", "gov", "regulator")):
            bonus += 0.2
    elif profile_id == "market_data":
        if row.get("symbol") or row.get("document_type") == "market_symbol":
            bonus += 0.2

    reliability = float(row.get("_reliability") or row.get("reliability_score") or 0.5)
    return min(1.0, base * 0.75 + bonus + reliability * 0.1)


def rank_rows(
    rows: list[dict[str, Any]],
    *,
    query: str,
    profile: str,
    max_results: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        title = str(row.get("title") or "").strip().lower()
        url = str(row.get("url") or "").strip().lower()
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        score = score_row(row, query=query, profile=profile)
        enriched = dict(row)
        enriched["_generic_relevance"] = score
        scored.append((score, enriched))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _score, row in scored[: max(0, max_results)]]
