"""
Memory v7.1 consolidation scoring — deterministic staging for review.
"""
from __future__ import annotations

import time
from typing import Any

CONSOLIDATION_STAGE_THRESHOLD = 0.55


def compute_consolidation_score(payload: dict, *, now: float | None = None) -> float:
    """0–1 score from cross-day retrieval, citations, and provenance."""
    from core.memory_promotion import (
        _avg_retrieval_score,
        _clamp01,
        consolidation_from_retrieval_days,
    )

    ts = float(now if now is not None else time.time())
    retrieval_days = list(payload.get("retrieval_days") or [])
    retrieved = max(0, int(payload.get("times_retrieved") or 0))
    cited = max(0, int(payload.get("times_cited_positively") or 0))

    multi_day = consolidation_from_retrieval_days(retrieval_days)
    citation = _clamp01(cited / max(retrieved, 1)) if retrieved else 0.0
    provenance = 1.0 if payload.get("provenance_quote") or payload.get("links_to_document_ids") else 0.3
    avg_score = _avg_retrieval_score(payload)

    score = 0.35 * multi_day + 0.25 * citation + 0.20 * provenance + 0.20 * avg_score
    if int(payload.get("times_episode_overlap") or 0) > 0:
        score += 0.05
    if int(payload.get("times_salvage_considered") or 0) > 0:
        score += 0.03
    return _clamp01(score)


def build_consolidation_hints(payload: dict, *, now: float | None = None) -> list[str]:
    hints: list[str] = []
    days = list(payload.get("retrieval_days") or [])
    if len(days) >= 2:
        hints.append("multi_day_retrieval")
    retrieved = int(payload.get("times_retrieved") or 0)
    cited = int(payload.get("times_cited_positively") or 0)
    if retrieved and cited / max(retrieved, 1) >= 0.5:
        hints.append("high_citation")
    if payload.get("provenance_quote") or payload.get("links_to_document_ids"):
        hints.append("provenance_present")
    if int(payload.get("times_episode_overlap") or 0) > 0:
        hints.append("episode_overlap")
    if int(payload.get("times_salvage_considered") or 0) > 0:
        hints.append("salvage_touch")
    return hints


def should_stage_for_consolidation(
    payload: dict,
    source: str,
    *,
    now: float | None = None,
) -> tuple[bool, float, list[str]]:
    src = (source or "").lower()
    if payload.get("promoted_at") or payload.get("flagged_for_review"):
        return False, 0.0, []
    if str(payload.get("category") or "").lower() == "episode":
        return False, 0.0, []
    if not any(src.startswith(p) for p in ("qube_memory::context::", "qube_memory::knowledge::", "qube_memory::legacy::")):
        return False, 0.0, []

    score = compute_consolidation_score(payload, now=now)
    hints = build_consolidation_hints(payload, now=now)
    return score >= CONSOLIDATION_STAGE_THRESHOLD, score, hints
