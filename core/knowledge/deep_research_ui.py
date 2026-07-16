"""UI helpers for deep-research progress and gating."""

from __future__ import annotations


def deep_research_progress_percent(payload: dict) -> int:
    """Map worker progress payloads to a 0–100 bar value."""
    phase = str(payload.get("phase") or "")
    if phase == "decomposing":
        return 12
    if phase == "retrieving":
        idx = max(0, int(payload.get("sub_query_index") or 0))
        total = max(1, int(payload.get("sub_query_total") or 1))
        return 15 + int(55 * idx / total)
    if phase == "merging":
        return 85
    if phase == "reporting":
        return 95
    if phase == "synthesizing":
        return 92
    return 0


def deep_research_available(*, enabled: bool) -> bool:
    return bool(enabled)
