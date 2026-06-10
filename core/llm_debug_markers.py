"""
Visual markers in logs/llm_debug.log to separate chat exchanges and inference calls.

Observer-only; no inference or prompt changes. Gated by QUBE_LLM_DEBUG (same as
core.native_llm_debug.llm_debug_enabled).
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

from core.native_llm_debug import llm_debug_enabled

logger = logging.getLogger("Qube.NativeLLM.Debug")

_MARKER_WIDTH = 78
_exchange_lock = threading.Lock()
_exchange_seq = 0


def next_exchange_id() -> int:
    global _exchange_seq
    with _exchange_lock:
        _exchange_seq += 1
        return _exchange_seq


def _preview(text: str, *, limit: int = 120) -> str:
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _banner_line(label: str) -> str:
    inner = f" {label} "
    pad = max(0, _MARKER_WIDTH - len(inner))
    left = pad // 2
    right = pad - left
    return "=" * left + inner + "=" * right


def log_chat_exchange_begin(
    *,
    exchange_id: int,
    session_id: str,
    user_prompt: str,
    engine_mode: str = "",
) -> None:
    if not llm_debug_enabled():
        return
    payload: dict[str, Any] = {
        "event": "llm_debug_exchange_begin",
        "exchange_id": exchange_id,
        "session_id": session_id or "",
        "engine_mode": engine_mode or "",
        "user_prompt_preview": _preview(user_prompt),
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
    logger.info(_banner_line(f"[QUBE CHAT EXCHANGE BEGIN] id={exchange_id}"))
    logger.info(
        "[QUBE EXCHANGE] session=%s engine=%s user=%r",
        session_id or "(none)",
        engine_mode or "?",
        _preview(user_prompt),
    )


def log_chat_exchange_end(
    *,
    exchange_id: int,
    session_id: str,
    route: str = "",
    success: bool = False,
    presented_text: str = "",
    worker_prep_ms: int | None = None,
    engine_queue_wait_ms: int | None = None,
    engine_inference_ms: int | None = None,
    exchange_total_ms: int | None = None,
) -> None:
    if not llm_debug_enabled():
        return
    presented_len = len(presented_text or "")
    payload: dict[str, Any] = {
        "event": "llm_debug_exchange_end",
        "exchange_id": exchange_id,
        "session_id": session_id or "",
        "route": route or "",
        "success": bool(success),
        "presented_len": presented_len,
        "presented_preview": _preview(presented_text),
    }
    if worker_prep_ms is not None:
        payload["worker_prep_ms"] = int(worker_prep_ms)
    if engine_queue_wait_ms is not None:
        payload["engine_queue_wait_ms"] = int(engine_queue_wait_ms)
    if engine_inference_ms is not None:
        payload["engine_inference_ms"] = int(engine_inference_ms)
    if exchange_total_ms is not None:
        payload["exchange_total_ms"] = int(exchange_total_ms)
    logger.info(json.dumps(payload, ensure_ascii=False))
    logger.info(
        "[QUBE EXCHANGE] session=%s route=%s success=%s presented_len=%d preview=%r",
        session_id or "(none)",
        route or "?",
        success,
        presented_len,
        _preview(presented_text),
    )
    logger.info(_banner_line(f"[QUBE CHAT EXCHANGE END] id={exchange_id}"))


def log_inference_scope_begin(
    *,
    caller: str,
    exchange_id: Optional[int] = None,
    stream: bool = False,
) -> None:
    if not llm_debug_enabled():
        return
    ex = f" exchange={exchange_id}" if exchange_id is not None else ""
    logger.info(
        "[QUBE INFERENCE BEGIN] caller=%s%s stream=%s",
        caller or "unknown",
        ex,
        stream,
    )


def log_inference_scope_end(
    *,
    caller: str,
    exchange_id: Optional[int] = None,
) -> None:
    if not llm_debug_enabled():
        return
    ex = f" exchange={exchange_id}" if exchange_id is not None else ""
    logger.info("[QUBE INFERENCE END] caller=%s%s", caller or "unknown", ex)


def log_inference_token_begin(
    *,
    caller: str,
    exchange_id: Optional[int] = None,
    stream: bool = False,
) -> None:
    """Emitted immediately before create_completion (true inference boundary)."""
    if not llm_debug_enabled():
        return
    ex = f" exchange={exchange_id}" if exchange_id is not None else ""
    logger.info(
        "[QUBE INFERENCE TOKEN BEGIN] caller=%s%s stream=%s",
        caller or "unknown",
        ex,
        stream,
    )


def log_inference_token_end(
    *,
    caller: str,
    exchange_id: Optional[int] = None,
) -> None:
    if not llm_debug_enabled():
        return
    ex = f" exchange={exchange_id}" if exchange_id is not None else ""
    logger.info("[QUBE INFERENCE TOKEN END] caller=%s%s", caller or "unknown", ex)


def log_engine_queue_snapshot(snapshot: dict[str, Any]) -> None:
    if not llm_debug_enabled():
        return
    payload = {"event": "llm_engine_queue_snapshot", **snapshot}
    logger.info(json.dumps(payload, ensure_ascii=False))


def log_engine_job_timing(timing_dict: dict[str, Any], *, background: bool = False) -> None:
    if not llm_debug_enabled():
        return
    event = "llm_engine_background_job_timing" if background else "llm_engine_job_timing"
    payload = {"event": event, **timing_dict}
    logger.info(json.dumps(payload, ensure_ascii=False))
