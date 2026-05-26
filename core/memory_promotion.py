"""
Memory v7.1 promotion scoring — Qube-tuned consolidation and exposure signals.
"""
from __future__ import annotations

import math
import time
from typing import Any

PROMOTION_WEIGHTS: dict[str, float] = {
    "relevance": 0.30,
    "frequency": 0.24,
    "query_diversity": 0.15,
    "recency": 0.15,
    "consolidation": 0.10,
    "richness": 0.06,
}

PROMOTION_CANDIDATE_MIN_SCORE = 0.65

PROMOTABLE_TIER_PREFIXES = (
    "qube_memory::context::",
    "qube_memory::knowledge::",
    "qube_memory::legacy::",
)

PROMOTION_PRESETS: dict[str, dict[str, float | int]] = {
    "conservative": {
        "min_score": 0.85,
        "min_retrieved": 4,
        "min_unique": 4,
        "max_age_days": 21,
    },
    "standard": {
        "min_score": 0.78,
        "min_retrieved": 3,
        "min_unique": 3,
        "max_age_days": 30,
    },
    "aggressive": {
        "min_score": 0.72,
        "min_retrieved": 2,
        "min_unique": 2,
        "max_age_days": 45,
    },
}

DEFAULT_PROMOTION_PRESET = "standard"
PROMOTION_RECENCY_HALF_LIFE_DAYS = 14.0
PROMOTION_NEAR_DUPLICATE_DISTANCE = 0.22


def get_promotion_thresholds(preset: str | None = None) -> dict[str, float | int]:
    key = (preset or DEFAULT_PROMOTION_PRESET).strip().lower()
    if key not in PROMOTION_PRESETS:
        key = DEFAULT_PROMOTION_PRESET
    return dict(PROMOTION_PRESETS[key])


def tier_from_source(source: str) -> str:
    src = (source or "").lower()
    for tier in ("preference", "knowledge", "episode", "context", "legacy"):
        if f"qube_memory::{tier}::" in src:
            return tier
    return "context"


def is_promotable_source(source: str) -> bool:
    src = (source or "").lower()
    return any(src.startswith(p) for p in PROMOTABLE_TIER_PREFIXES)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _recency_score(last_used_at: float, now: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    age_days = max(0.0, (now - last_used_at) / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def _exposure_count(payload: dict) -> int:
    retrieved = max(0, int(payload.get("times_retrieved") or 0))
    cited = min(max(0, int(payload.get("times_cited_positively") or 0)), 3)
    salvage = min(max(0, int(payload.get("times_salvage_considered") or 0)), 2)
    return retrieved + cited + salvage


def _avg_retrieval_score(payload: dict) -> float:
    count = int(payload.get("retrieval_score_count") or 0)
    if count <= 0:
        return 0.0
    total = float(payload.get("retrieval_score_sum") or 0.0)
    return _clamp01(total / max(1, count))


def consolidation_from_retrieval_days(retrieval_days: list[str]) -> float:
    """Multi-day consolidation score from ISO day buckets."""
    from datetime import datetime, timezone

    days = [str(d) for d in (retrieval_days or []) if d]
    if not days:
        return 0.0
    if len(days) == 1:
        return 0.2
    parsed: list[float] = []
    for day in days[:16]:
        try:
            dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            parsed.append(dt.timestamp())
        except ValueError:
            continue
    parsed.sort()
    if len(parsed) <= 1:
        return 0.2
    span_days = max(0.0, (parsed[-1] - parsed[0]) / 86400.0)
    spacing = _clamp01(math.log1p(len(parsed) - 1) / math.log1p(4))
    span = _clamp01(span_days / 7.0)
    return _clamp01(0.55 * spacing + 0.45 * span)


def _richness_score(payload: dict) -> float:
    content = (payload.get("content") or "").strip()
    if len(content.split()) < 4:
        return 0.0
    if payload.get("provenance_quote") or payload.get("links_to_document_ids"):
        return 1.0
    return 0.5


def compute_promotion_signals(payload: dict, *, now: float | None = None) -> dict[str, float]:
    ts = float(now if now is not None else time.time())
    exposure = _exposure_count(payload)
    retrieved = max(0, int(payload.get("times_retrieved") or 0))
    cited = max(0, int(payload.get("times_cited_positively") or 0))
    unique_q = max(0, int(payload.get("unique_query_count") or 0))
    if unique_q == 0 and payload.get("retrieval_query_fps"):
        unique_q = len(set(payload.get("retrieval_query_fps") or []))
    retrieval_days = list(payload.get("retrieval_days") or [])
    context_diversity = max(unique_q, len(retrieval_days))

    last_used = payload.get("last_used_at") or payload.get("timestamp") or ts
    try:
        last_used = float(last_used)
    except (TypeError, ValueError):
        last_used = ts

    citation_rate = _clamp01(cited / max(retrieved, 1))
    avg_score = _avg_retrieval_score(payload)
    if avg_score > 0:
        relevance = _clamp01(0.7 * citation_rate + 0.3 * avg_score)
    else:
        relevance = citation_rate

    frequency = _clamp01(math.log1p(exposure) / math.log1p(10))
    query_diversity = _clamp01(context_diversity / 5.0)
    recency = _recency_score(last_used, ts, PROMOTION_RECENCY_HALF_LIFE_DAYS)
    consolidation = max(
        consolidation_from_retrieval_days(retrieval_days),
        _clamp01(int(payload.get("times_episode_overlap") or 0) / 3.0),
    )
    richness = _richness_score(payload)

    return {
        "relevance": relevance,
        "frequency": frequency,
        "query_diversity": query_diversity,
        "recency": recency,
        "consolidation": consolidation,
        "richness": richness,
    }


def compute_promotion_score(payload: dict, *, now: float | None = None) -> float:
    signals = compute_promotion_signals(payload, now=now)
    total = sum(PROMOTION_WEIGHTS[k] * signals.get(k, 0.0) for k in PROMOTION_WEIGHTS)
    boost = _clamp01(float(payload.get("consolidation_score") or 0.0) * 0.05)
    return _clamp01(total + boost)


def promotion_score_breakdown(payload: dict, *, now: float | None = None) -> list[dict[str, Any]]:
    signals = compute_promotion_signals(payload, now=now)
    score = compute_promotion_score(payload, now=now)
    rows: list[dict[str, Any]] = []
    for key, weight in PROMOTION_WEIGHTS.items():
        raw = signals.get(key, 0.0)
        rows.append(
            {
                "signal": key,
                "raw": round(raw, 4),
                "weight": weight,
                "contribution": round(weight * raw, 4),
            }
        )
    rows.append({"signal": "total", "raw": round(score, 4), "weight": 1.0, "contribution": round(score, 4)})
    return rows


def passes_promotion_gates(
    payload: dict,
    source: str,
    *,
    now: float | None = None,
    preset: str | None = None,
) -> tuple[bool, str]:
    ok, reason, _ = passes_promotion_gates_with_reason(payload, source, now=now, preset=preset)
    return ok, reason


def passes_promotion_gates_with_reason(
    payload: dict,
    source: str,
    *,
    now: float | None = None,
    preset: str | None = None,
) -> tuple[bool, str, dict[str, float]]:
    ts = float(now if now is not None else time.time())
    thresholds = get_promotion_thresholds(preset)
    components = compute_promotion_signals(payload, now=ts)

    if not is_promotable_source(source):
        return False, "not_promotable_tier", components
    if str(payload.get("category") or "").lower() == "episode":
        return False, "episode_row", components

    retrieved = int(payload.get("times_retrieved") or 0)
    unique_q = int(payload.get("unique_query_count") or 0)
    if unique_q == 0 and payload.get("retrieval_query_fps"):
        unique_q = len(set(payload.get("retrieval_query_fps") or []))
    retrieval_days = list(payload.get("retrieval_days") or [])
    context_diversity = max(unique_q, len(retrieval_days))

    created = payload.get("first_seen_at") or payload.get("timestamp") or ts
    try:
        age_days = max(0.0, (ts - float(created)) / 86400.0)
    except (TypeError, ValueError):
        age_days = 0.0

    min_retrieved = int(thresholds["min_retrieved"])
    min_unique = int(thresholds["min_unique"])
    max_age = int(thresholds["max_age_days"])
    min_score = float(thresholds["min_score"])

    if retrieved < min_retrieved:
        return False, f"times_retrieved {retrieved} < {min_retrieved}", components
    if context_diversity < min_unique:
        return False, f"unique_context {context_diversity} < {min_unique}", components
    if age_days > max_age:
        return False, f"age_days {age_days:.1f} > {max_age}", components

    score = compute_promotion_score(payload, now=ts)
    if score < min_score:
        return False, f"score {score:.3f} < {min_score}", components
    return True, "ok", components


def is_promotion_candidate(payload: dict, *, now: float | None = None) -> bool:
    return compute_promotion_score(payload, now=now) >= PROMOTION_CANDIDATE_MIN_SCORE


def is_almost_promoted(payload: dict, source: str, *, now: float | None = None, preset: str | None = None) -> bool:
    """Score meets candidate bar but a gate still fails."""
    if payload.get("promoted_at"):
        return False
    if not is_promotion_candidate(payload, now=now):
        return False
    ok, _reason, _ = passes_promotion_gates_with_reason(payload, source, now=now, preset=preset)
    return not ok
