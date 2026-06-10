"""Collapse diagnostics observability for llm_debug.log."""
from __future__ import annotations

from typing import Any

from core.llm_structured_log import structured_llm_log


def log_collapse_diagnostics(
    *,
    session_id: str = "",
    turn_index: int = 0,
    collapse_risk: str = "",
    collapse_score: float = 0.0,
    prompt_length: int = 0,
    output_length: int = 0,
    rewrite_confidence: float = 0.0,
    degeneration_score: float = 0.0,
    hallucination_score: float = 0.0,
    format_drift_score: float = 0.0,
    hallucination_flags: list[str] | tuple[str, ...] = (),
    format_drift_flags: list[str] | tuple[str, ...] = (),
    prior_turn_suppressed: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "collapse_turn_index": turn_index,
        "collapse_risk": collapse_risk,
        "collapse_score": round(collapse_score, 3),
        "collapse_prompt_length": prompt_length,
        "collapse_output_length": output_length,
        "collapse_rewrite_confidence": round(rewrite_confidence, 3),
        "collapse_degeneration_score": round(degeneration_score, 3),
        "collapse_hallucination_score": round(hallucination_score, 3),
        "collapse_format_drift_score": round(format_drift_score, 3),
        "collapse_hallucination_flags": list(hallucination_flags),
        "collapse_format_drift_flags": list(format_drift_flags),
        "prior_turn_suppressed": bool(prior_turn_suppressed),
    }
    if extra:
        payload.update(extra)
    structured_llm_log("collapse_diagnostics", payload)
