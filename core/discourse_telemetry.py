"""Structured discourse observability events for llm_debug.log."""
from __future__ import annotations

from typing import Any, Optional

from core.llm_structured_log import structured_llm_log


def log_discourse_referent_trace(
    *,
    referent: str,
    referent_source: str,
    referent_confidence: float,
    user_prompt_preview: str = "",
    assistant_preview: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "referent": referent,
        "referent_source": referent_source,
        "referent_confidence": round(referent_confidence, 3),
        "user_prompt_preview": (user_prompt_preview or "")[:200],
        "assistant_preview": (assistant_preview or "")[:200],
    }
    if extra:
        payload.update(extra)
    structured_llm_log("discourse_referent_trace", payload)


def log_discourse_query_rewrite(
    *,
    original: str,
    resolved: str,
    substitutions: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    confidence: float,
    rewrite_reason: str = "",
) -> None:
    structured_llm_log(
        "discourse_query_rewrite",
        {
            "original": (original or "")[:300],
            "resolved": (resolved or "")[:300],
            "substitutions": [list(pair) for pair in substitutions],
            "confidence": round(confidence, 3),
            "rewrite_reason": rewrite_reason,
        },
    )


def log_discourse_referent_rejected(
    *,
    candidate: str,
    candidate_source: str,
    reject_reason: str,
    prior_referent: str = "",
    user_prompt_preview: str = "",
    assistant_preview: str = "",
) -> None:
    structured_llm_log(
        "discourse_referent_rejected",
        {
            "candidate": (candidate or "")[:120],
            "candidate_source": candidate_source or "",
            "reject_reason": reject_reason or "",
            "prior_referent": (prior_referent or "")[:120],
            "user_prompt_preview": (user_prompt_preview or "")[:200],
            "assistant_preview": (assistant_preview or "")[:200],
        },
    )


def log_discourse_rewrite_validation_failed(
    *,
    original: str,
    resolved: str,
    reject_reason: str,
    referent: str = "",
) -> None:
    structured_llm_log(
        "discourse_rewrite_validation_failed",
        {
            "original": (original or "")[:300],
            "resolved": (resolved or "")[:300],
            "reject_reason": reject_reason or "",
            "referent": (referent or "")[:120],
        },
    )


def log_discourse_prompt_rewrite(
    *,
    original: str,
    grounded: str,
    rewrite_anchor: str = "",
    rewrite_confidence: float = 0.0,
    rewrite_reason: str = "",
    applied: bool = False,
    salience_anchor: str = "",
    salience_reason: str = "",
) -> None:
    structured_llm_log(
        "discourse_prompt_rewrite",
        {
            "original": (original or "")[:300],
            "grounded": (grounded or "")[:300],
            "rewrite_anchor": rewrite_anchor or "",
            "rewrite_confidence": round(rewrite_confidence, 3),
            "rewrite_reason": rewrite_reason,
            "applied": bool(applied),
            "salience_anchor": salience_anchor or "",
            "salience_reason": salience_reason or "",
        },
    )
