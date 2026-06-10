"""Structured discourse observability events for llm_debug.log."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

_logger = logging.getLogger("Qube.NativeLLM.Debug")


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
        "event": "discourse_referent_trace",
        "referent": referent,
        "referent_source": referent_source,
        "referent_confidence": round(referent_confidence, 3),
        "user_prompt_preview": (user_prompt_preview or "")[:200],
        "assistant_preview": (assistant_preview or "")[:200],
    }
    if extra:
        payload.update(extra)
    _logger.info(json.dumps(payload, ensure_ascii=False))


def log_discourse_query_rewrite(
    *,
    original: str,
    resolved: str,
    substitutions: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    confidence: float,
    rewrite_reason: str = "",
) -> None:
    payload = {
        "event": "discourse_query_rewrite",
        "original": (original or "")[:300],
        "resolved": (resolved or "")[:300],
        "substitutions": [list(pair) for pair in substitutions],
        "confidence": round(confidence, 3),
        "rewrite_reason": rewrite_reason,
    }
    _logger.info(json.dumps(payload, ensure_ascii=False))
