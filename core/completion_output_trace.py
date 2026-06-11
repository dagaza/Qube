"""
Observer-only completion output trace: raw model text vs worker-filtered vs UI-presented.

Enable with QUBE_LOG_RAW_COMPLETION=1 (logs to Qube.NativeLLM.Debug -> ~/.qube/logs/llm_debug.log).

Optional: QUBE_LOG_RAW_COMPLETION_MAX_CHARS=N (default 0 = no truncation per field).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("Qube.NativeLLM.Debug")

_EVENT = "llm_completion_output_trace"

_VALIDATION_TRACE_KEYS: tuple[str, ...] = (
    "first_pass_raw_len",
    "sanitized_len",
    "validation_issues",
    "validation_severity",
    "raw_validation_issues",
    "raw_validation_severity",
    "retry_attempted",
    "retry_used",
    "retry_reason",
    "retry_len",
    "retry_max_tokens",
    "original_max_tokens",
    "original_format",
    "final_format",
    "replacement_suppressed",
    "replacement_rejection_reason",
    "streamed_visible_len",
    "retry_replaced_stream",
    "post_inference_ms",
)


def completion_output_trace_enabled() -> bool:
    return os.environ.get("QUBE_LOG_RAW_COMPLETION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def completion_output_trace_max_chars() -> int:
    raw = (os.environ.get("QUBE_LOG_RAW_COMPLETION_MAX_CHARS") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


@dataclass
class CompletionOutputSnapshot:
    """Captured at end of worker streaming; ``presented_text`` is filled in ``run()``."""

    engine_mode: str
    raw_text: str = ""
    after_harmony_parser: str = ""
    after_worker_filters: str = ""
    streamed_incremental: str = ""
    worker_return_text: str = ""
    engine_end_text: str = ""
    retry_replaced: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _truncate_field(text: str, max_chars: int) -> tuple[str, Optional[int]]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, None
    return text[:max_chars], len(text)


def _stages_changed(values: dict[str, str]) -> list[str]:
    """Names of stages whose text differs from the previous stage."""
    order = (
        "raw_text",
        "after_harmony_parser",
        "after_worker_filters",
        "streamed_incremental",
        "worker_return_text",
        "presented_text",
    )
    changed: list[str] = []
    prev: str | None = None
    prev_name: str | None = None
    for name in order:
        cur = values.get(name) or ""
        if prev is not None and cur != prev:
            changed.append(f"{prev_name}->{name}")
        prev = cur
        prev_name = name
    return changed


def build_completion_output_trace_payload(
    *,
    session_id: str,
    snapshot: CompletionOutputSnapshot,
    presented_text: str,
) -> dict[str, Any]:
    max_chars = completion_output_trace_max_chars()
    values = {
        "raw_text": snapshot.raw_text or "",
        "after_harmony_parser": snapshot.after_harmony_parser or "",
        "after_worker_filters": snapshot.after_worker_filters or "",
        "streamed_incremental": snapshot.streamed_incremental or "",
        "worker_return_text": snapshot.worker_return_text or "",
        "presented_text": presented_text or "",
    }
    text_fields: dict[str, Any] = {}
    full_lengths: dict[str, int] = {}
    truncated_any = False
    for key, value in values.items():
        stored, full_len = _truncate_field(value, max_chars)
        text_fields[key] = stored
        if full_len is not None:
            truncated_any = True
            full_lengths[f"{key}_full_len"] = full_len
            text_fields[f"{key}_truncated"] = True

    payload: dict[str, Any] = {
        "event": _EVENT,
        "session_id": session_id or "",
        "engine_mode": snapshot.engine_mode or "",
        "retry_replaced": bool(snapshot.retry_replaced),
        **text_fields,
        "raw_len": len(snapshot.raw_text or ""),
        "after_harmony_parser_len": len(snapshot.after_harmony_parser or ""),
        "after_worker_filters_len": len(snapshot.after_worker_filters or ""),
        "streamed_incremental_len": len(snapshot.streamed_incremental or ""),
        "worker_return_len": len(snapshot.worker_return_text or ""),
        "presented_len": len(presented_text or ""),
        "removed_char_count": max(
            0, len(snapshot.raw_text or "") - len(presented_text or "")
        ),
        "raw_equals_presented": (snapshot.raw_text or "") == (presented_text or ""),
        "stream_incremental_diverged_from_filters": (
            (snapshot.streamed_incremental or "")
            != (snapshot.after_worker_filters or "")
        ),
        "stages_changed": _stages_changed(values),
    }
    if truncated_any:
        payload["truncated"] = True
        payload.update(full_lengths)
    if snapshot.extra:
        payload["extra"] = dict(snapshot.extra)
        for key in _VALIDATION_TRACE_KEYS:
            if key in snapshot.extra:
                payload[key] = snapshot.extra[key]
    return payload


def log_completion_output_trace(
    *,
    session_id: str,
    snapshot: Optional[CompletionOutputSnapshot],
    presented_text: str,
) -> None:
    if not completion_output_trace_enabled() or snapshot is None:
        return
    payload = build_completion_output_trace_payload(
        session_id=session_id,
        snapshot=snapshot,
        presented_text=presented_text,
    )
    logger.info(json.dumps(payload, ensure_ascii=False))
    logger.info(
        "[CompletionOutputTrace] session=%s engine=%s raw_len=%d presented_len=%d "
        "removed=%d stages_changed=%s",
        payload.get("session_id") or "(none)",
        payload.get("engine_mode") or "?",
        payload.get("raw_len", 0),
        payload.get("presented_len", 0),
        payload.get("removed_char_count", 0),
        payload.get("stages_changed") or [],
    )
