"""Structured history degeneration observability for llm_debug.log."""
from __future__ import annotations

from core.llm_structured_log import structured_llm_log


def log_history_degeneration_suppression(
    *,
    session_id: str = "",
    score: float,
    flags: list[str] | tuple[str, ...],
    presented_preview: str = "",
    stored_content: str = "",
    output_degeneration: dict | None = None,
    stream_cancelled: bool = False,
    suppression_reason: str = "pathology",
) -> None:
    payload: dict = {
        "session_id": session_id,
        "history_degeneration_score": round(score, 3),
        "history_degeneration_flags": list(flags),
        "history_degeneration_suppressed": True,
        "presented_preview": (presented_preview or "")[:300],
        "stored_content": (stored_content or "")[:200],
        "stream_cancelled": bool(stream_cancelled),
        "suppression_reason": suppression_reason,
    }
    if output_degeneration:
        payload.update(output_degeneration)
    structured_llm_log("history_degeneration_suppression", payload)
