"""
Memory v7 retrieval policy helpers — core-memory injection gate, MMR, decay.
"""
from __future__ import annotations

import difflib
import math
import re
import time
from typing import Any


CORE_MEMORY_MIN_SCORE = 0.45
CORE_MEMORY_MIN_MARGIN = 0.08

HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_TEXT_WEIGHT = 0.3
HYBRID_CANDIDATE_MULTIPLIER = 4

MMR_LAMBDA = 0.7

TEMPORAL_HALF_LIFE_DAYS: dict[str, float | None] = {
    "preference": None,
    "knowledge": 30.0,
    "context": 30.0,
    "episode": 14.0,
    "legacy": 30.0,
}


def apply_core_memory_gate(scored_items: list[dict]) -> list[dict]:
    """Suppress weak core-memory hits on plain CHAT turns."""
    if not scored_items:
        return []
    top = scored_items[0]
    top_score = float(top.get("score") or 0.0)
    if top_score < CORE_MEMORY_MIN_SCORE:
        return []
    if len(scored_items) >= 2:
        second = float(scored_items[1].get("score") or 0.0)
        if (top_score - second) < CORE_MEMORY_MIN_MARGIN:
            return []
    return scored_items


def temporal_decay_multiplier(tier: str, payload: dict, now: float | None = None) -> float:
    """Evergreen tiers (preference) skip decay; others use half-life from last use."""
    half_life = TEMPORAL_HALF_LIFE_DAYS.get(str(tier or "context").lower(), 30.0)
    if half_life is None:
        return 1.0
    ts = float(now if now is not None else time.time())
    last_used = payload.get("last_used_at") or payload.get("timestamp") or ts
    try:
        last_used = float(last_used)
    except (TypeError, ValueError):
        last_used = ts
    age_days = max(0.0, (ts - last_used) / 86400.0)
    return 0.5 ** (age_days / half_life)


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-z0-9]{2,}", (text or "").lower())}


def apply_mmr(items: list[dict], *, lambda_: float = MMR_LAMBDA, top_k: int = 5) -> list[dict]:
    """Maximal Marginal Relevance rerank on memory content strings."""
    if len(items) <= 1:
        return items[:top_k]

    raw_scores = [float(c.get("score") or 0.0) for c in items]
    lo = min(raw_scores)
    hi = max(raw_scores)
    span = hi - lo

    def _norm_rel(idx: int) -> float:
        if span <= 1e-9:
            return 1.0 if raw_scores[idx] > 0 else 0.0
        return (raw_scores[idx] - lo) / span

    selected: list[dict] = []
    remaining = list(items)
    while remaining and len(selected) < top_k:
        ranked: list[tuple[int, float, float]] = []
        for idx, cand in enumerate(remaining):
            rel = _norm_rel(idx)
            cand_text = (cand.get("content") or "").strip().lower()
            max_sim = 0.0
            for picked in selected:
                picked_text = (picked.get("content") or "").strip().lower()
                if not cand_text or not picked_text:
                    continue
                max_sim = max(
                    max_sim,
                    difflib.SequenceMatcher(None, cand_text, picked_text).ratio(),
                )
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            ranked.append((idx, mmr, max_sim))
        ranked.sort(key=lambda item: item[1], reverse=True)
        pick_idx = ranked[0][0]
        for idx, _mmr, max_sim in ranked:
            if not selected or max_sim < 0.85:
                pick_idx = idx
                break
        selected.append(remaining.pop(pick_idx))
    return selected


def fts_query_token_overlap(query: str, content: str, min_ratio: float = 0.25) -> bool:
    """Lightweight gate for FTS-only memory hits."""
    q_tokens = _token_set(query)
    c_tokens = _token_set(content)
    if not q_tokens:
        return True
    if not c_tokens:
        return False
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    return overlap >= min_ratio


def tier_from_source(source: str) -> str:
    src = (source or "").lower()
    for tier in ("preference", "knowledge", "episode", "context", "legacy"):
        if f"qube_memory::{tier}::" in src:
            return tier
    return "context"
