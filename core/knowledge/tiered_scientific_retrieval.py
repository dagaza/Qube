"""Tiered scientific adapter fan-out (HTTP resilience Slice 8)."""

from __future__ import annotations

import os
from typing import Any

from core.knowledge.scientific_discipline_packs import get_discipline_pack


def tiered_scientific_retrieval_enabled() -> bool:
    raw = os.getenv("QUBE_TIERED_SCIENTIFIC_RETRIEVAL")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def tiered_fallback_threshold(*, budget: int) -> int:
    """Minimum raw candidate rows before skipping fallback adapters."""
    raw = os.getenv("QUBE_TIERED_SCIENTIFIC_THRESHOLD")
    if raw is not None:
        try:
            return max(1, int(str(raw).strip()))
        except ValueError:
            pass
    return max(2, budget)


def split_adapter_tiers(
    adapter_ids: tuple[str, ...],
    *,
    discipline: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split resolved adapters into phase-1 primaries and phase-2 fallbacks."""
    if not adapter_ids:
        return (), ()

    pack = get_discipline_pack(discipline)
    primary_ids = frozenset(pack.primary_adapters) if pack is not None else frozenset()

    primary = tuple(aid for aid in adapter_ids if aid in primary_ids)
    fallback = tuple(aid for aid in adapter_ids if aid not in primary_ids)

    if not primary:
        return (adapter_ids[0],), adapter_ids[1:]

    return primary, fallback


def tiered_retrieval_diag(
    *,
    enabled: bool,
    primary: tuple[str, ...],
    fallback: tuple[str, ...],
    phase2_invoked: bool,
    threshold: int,
    candidate_count: int,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "threshold": threshold,
        "phase1_adapters": list(primary),
        "phase2_adapters": list(fallback) if phase2_invoked else [],
        "phase2_skipped": bool(fallback) and not phase2_invoked,
        "candidate_count_after_phase1": candidate_count,
    }
