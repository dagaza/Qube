"""Cross-source reliability scoring for evidence bundles."""

from __future__ import annotations

import difflib
import re
from typing import Any

_POSITIVE = re.compile(
    r"\b(effective|benefit|beneficial|improves|reduces|safe|successful|significant)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(ineffective|no benefit|no significant|harmful|adverse|failed|worsens)\b",
    re.IGNORECASE,
)


def _stance(text: str) -> str:
    if _NEGATIVE.search(text):
        return "negative"
    if _POSITIVE.search(text):
        return "positive"
    return "neutral"


def reliability_score_for_row(
    row: dict[str, Any],
    *,
    peer_rows: list[dict[str, Any]],
) -> float:
    """Heuristic agreement score from stance alignment with peer excerpts."""
    text = f"{row.get('title') or ''} {row.get('snippet') or ''}"
    stance = _stance(text)
    if not peer_rows:
        return 0.55

    agreements = 0
    comparisons = 0
    for peer in peer_rows:
        if peer is row:
            continue
        peer_text = f"{peer.get('title') or ''} {peer.get('snippet') or ''}"
        peer_stance = _stance(peer_text)
        comparisons += 1
        if stance == peer_stance and stance != "neutral":
            agreements += 1
        elif stance != "neutral" and peer_stance != "neutral" and stance != peer_stance:
            agreements -= 1
        sim = difflib.SequenceMatcher(None, text[:240], peer_text[:240]).ratio()
        if sim >= 0.55:
            agreements += 1

    if comparisons <= 0:
        return 0.55
    raw = 0.5 + (agreements / max(comparisons, 1)) * 0.35
    rel = float(row.get("_scientific_relevance") or row.get("_web_token_overlap") or 0.5)
    return max(0.0, min(0.95, raw * 0.7 + rel * 0.3))


def apply_reliability_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["_reliability_score"] = reliability_score_for_row(copy, peer_rows=rows)
        updated.append(copy)
    return updated
