"""
Structured 3-layer truth diff logging for LLM calls (L1 request, L2 prompt, L3 I/O).

Enable with ENABLE_LLM_TRUTH_DIFF_LOGGING=1 (logs to Qube.NativeLLM.Debug -> logs/llm_debug.log).

Optional: LLM_TRUTH_DIFF_MAX_CHARS=N (default 20000) truncates long text / string fields.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.canonical_fingerprint import (
    fingerprint_canonical_request,
    fingerprint_text,
    fingerprint_trace_component,
)
from core.canonical_request import CanonicalRequestExporter

logger = logging.getLogger("Qube.NativeLLM.Debug")

_DEFAULT_MAX_CHARS = 20_000

# Optional worker-side hooks so NativeLlamaEngine truth-diff events run through
# LLMWorker safe wrappers (same thread as inference; observer-only, non-blocking).
_l1_engine_hook: Optional[Callable[[dict, dict], None]] = None
_l2_prompt_hook: Optional[Callable[[str, dict], None]] = None


def bind_llm_worker_truth_diff_hooks(
    *,
    l1_engine_request: Optional[Callable[[dict, dict], None]] = None,
    l2_prompt: Optional[Callable[[str, dict], None]] = None,
) -> None:
    """Register LLMWorker safe emitters for native-engine L1/L2 (optional)."""
    global _l1_engine_hook, _l2_prompt_hook
    _l1_engine_hook = l1_engine_request
    _l2_prompt_hook = l2_prompt


def clear_llm_worker_truth_diff_hooks() -> None:
    bind_llm_worker_truth_diff_hooks()


def emit_l1_engine_request(request: dict, context: dict) -> None:
    if _l1_engine_hook is not None:
        try:
            _l1_engine_hook(request, context)
            return
        except Exception:
            logger.debug("[LLMTruthDiff] worker L1 engine hook failed", exc_info=True)
    get_llm_truth_diff_logger().log_l1_engine_request(request, context)


def emit_l2_prompt(prompt: str, metadata: dict) -> None:
    if _l2_prompt_hook is not None:
        try:
            _l2_prompt_hook(prompt, metadata)
            return
        except Exception:
            logger.debug("[LLMTruthDiff] worker L2 hook failed", exc_info=True)
    get_llm_truth_diff_logger().log_l2_prompt(prompt, metadata)


def llm_truth_diff_enabled() -> bool:
    return os.environ.get("ENABLE_LLM_TRUTH_DIFF_LOGGING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def llm_truth_diff_max_chars() -> int:
    raw = (os.environ.get("LLM_TRUTH_DIFF_MAX_CHARS") or "").strip()
    if not raw:
        return _DEFAULT_MAX_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_CHARS


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool, Optional[int]]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False, None
    return text[:max_chars], True, len(text)


def _sanitize_request_value(value: Any, max_chars: int, *, depth: int = 0) -> Any:
    """Deep-copy request payloads with long strings truncated."""
    if depth > 12:
        return "<max_depth>"
    if isinstance(value, str):
        stored, truncated, full_len = _truncate_text(value, max_chars)
        if truncated:
            return {"_truncated": True, "_full_len": full_len, "_preview": stored}
        return stored
    if isinstance(value, dict):
        return {
            str(k): _sanitize_request_value(v, max_chars, depth=depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_request_value(v, max_chars, depth=depth + 1) for v in value]
    return value


def _base_fields(
    *,
    layer: str,
    pipeline_stage: str,
    context: dict[str, Any],
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    request_id = context.get("request_id", context.get("exchange_id"))
    return {
        "layer": layer,
        "pipeline_stage": pipeline_stage,
        "timestamp": _utc_timestamp(),
        "request_id": request_id,
        "exchange_id": context.get("exchange_id", request_id),
        "session_id": str(context.get("session_id") or ""),
        "model_name": str(context.get("model_name") or ""),
        "fingerprint": dict(fingerprint),
        # Legacy aliases for grep-friendly log tooling.
        "content_hash": fingerprint.get("sha256", ""),
        "content_len": fingerprint.get("length", 0),
    }


class LLMTruthDiffLogger:
    """Observer-only structured logger for LLM request / prompt / output truth diffs."""

    def __init__(self, *, enabled: Optional[bool] = None) -> None:
        # None -> follow ENABLE_LLM_TRUTH_DIFF_LOGGING at call time (tests may pass True/False).
        self._enabled_override = enabled

    @property
    def enabled(self) -> bool:
        return self._is_active()

    def _is_active(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return llm_truth_diff_enabled()

    def _emit(self, payload: dict[str, Any]) -> None:
        if not self._is_active():
            return
        try:
            wrapped = {"llm_truth_diff": payload}
            logger.info(json.dumps(wrapped, ensure_ascii=False))
            logger.info(
                "[LLMTruthDiff] layer=%s stage=%s request_id=%s session=%s hash=%s…",
                payload.get("layer"),
                payload.get("pipeline_stage"),
                payload.get("request_id"),
                payload.get("session_id") or "(none)",
                str((payload.get("fingerprint") or {}).get("short") or "")[:12],
            )
        except Exception:
            logger.debug("[LLMTruthDiff] emit failed", exc_info=True)

    def log_l1_raw_request(self, request: dict, context: dict) -> None:
        """L1: frontend JSON before any worker-side processing."""
        if not self._is_active():
            return
        try:
            max_chars = llm_truth_diff_max_chars()
            sanitized = _sanitize_request_value(copy.deepcopy(request or {}), max_chars)
            fp = fingerprint_trace_component(request or {})
            payload = {
                **_base_fields(
                    layer="L1",
                    pipeline_stage="raw_request",
                    context=context or {},
                    fingerprint=fp,
                ),
                "request": sanitized,
            }
            if isinstance(sanitized, dict) and any(
                isinstance(v, dict) and v.get("_truncated") for v in sanitized.values()
            ):
                payload["truncated"] = True
            self._emit(payload)
        except Exception:
            logger.debug("[LLMTruthDiff] log_l1_raw_request failed", exc_info=True)

    def log_l1_engine_request(self, request: dict, context: dict) -> None:
        """L1: final JSON payload sent to the model backend."""
        if not self._is_active():
            return
        try:
            max_chars = llm_truth_diff_max_chars()
            sanitized = _sanitize_request_value(copy.deepcopy(request or {}), max_chars)
            canonical = CanonicalRequestExporter.export_canonical_request(request or {})
            fp = fingerprint_canonical_request(canonical)
            payload = {
                **_base_fields(
                    layer="L1",
                    pipeline_stage="engine_request",
                    context=context or {},
                    fingerprint=fp,
                ),
                "request": sanitized,
                "fingerprint_raw": fingerprint_trace_component(request or {}),
            }
            self._emit(payload)
        except Exception:
            logger.debug("[LLMTruthDiff] log_l1_engine_request failed", exc_info=True)

    def log_l2_prompt(self, prompt: str, metadata: dict) -> None:
        """L2: final rendered prompt string and template metadata."""
        if not self._is_active():
            return
        try:
            max_chars = llm_truth_diff_max_chars()
            text = prompt or ""
            stored, truncated, full_len = _truncate_text(text, max_chars)
            fp = fingerprint_text(text)
            payload: dict[str, Any] = {
                **_base_fields(
                    layer="L2",
                    pipeline_stage="prompt_render",
                    context=metadata or {},
                    fingerprint=fp,
                ),
                "prompt": stored,
                "template_source": str((metadata or {}).get("template_source") or ""),
                "chat_format_mode": str((metadata or {}).get("chat_format_mode") or ""),
                "execution_mode": str((metadata or {}).get("execution_mode") or ""),
                "prompt_contract_mode": str((metadata or {}).get("prompt_contract_mode") or ""),
            }
            if truncated:
                payload["truncated"] = True
                payload["prompt_full_len"] = full_len
            extra = {
                k: v
                for k, v in (metadata or {}).items()
                if k
                not in (
                    "template_source",
                    "chat_format_mode",
                    "execution_mode",
                    "prompt_contract_mode",
                    "request_id",
                    "exchange_id",
                    "session_id",
                    "model_name",
                )
            }
            if extra:
                payload["metadata"] = extra
            self._emit(payload)
        except Exception:
            logger.debug("[LLMTruthDiff] log_l2_prompt failed", exc_info=True)

    def log_l3_model_io(
        self,
        raw: str,
        after_stages: list[str],
        final: str,
        metadata: dict,
    ) -> None:
        """L3: raw model output, per-stage snapshots, and final presented text."""
        if not self._is_active():
            return
        try:
            max_chars = llm_truth_diff_max_chars()
            raw_text = raw or ""
            final_text = final or ""
            raw_fp = fingerprint_text(raw_text)
            final_fp = fingerprint_text(final_text)
            raw_stored, raw_trunc, raw_full = _truncate_text(raw_text, max_chars)
            final_stored, final_trunc, final_full = _truncate_text(final_text, max_chars)

            stages_payload: list[dict[str, Any]] = []
            for idx, stage_text in enumerate(after_stages or []):
                stage = stage_text or ""
                stored, truncated, full_len = _truncate_text(stage, max_chars)
                stage_fp = fingerprint_text(stage)
                entry: dict[str, Any] = {
                    "index": idx,
                    "stage": f"after_stage_{idx}",
                    "text": stored,
                    "fingerprint": stage_fp,
                    "content_hash": stage_fp["sha256"],
                    "content_len": stage_fp["length"],
                }
                if truncated:
                    entry["truncated"] = True
                    entry["text_full_len"] = full_len
                stages_payload.append(entry)

            payload: dict[str, Any] = {
                **_base_fields(
                    layer="L3",
                    pipeline_stage="model_io",
                    context=metadata or {},
                    fingerprint=raw_fp,
                ),
                "raw": raw_stored,
                "raw_len": len(raw_text),
                "raw_fingerprint": raw_fp,
                "after_stages": stages_payload,
                "final": final_stored,
                "final_len": len(final_text),
                "final_fingerprint": final_fp,
                "final_hash": final_fp["sha256"],
                "raw_equals_final": raw_text == final_text,
            }
            if raw_trunc:
                payload["raw_truncated"] = True
                payload["raw_full_len"] = raw_full
            if final_trunc:
                payload["final_truncated"] = True
                payload["final_full_len"] = final_full
            extra = {
                k: v
                for k, v in (metadata or {}).items()
                if k
                not in (
                    "request_id",
                    "exchange_id",
                    "session_id",
                    "model_name",
                )
            }
            if extra:
                payload["metadata"] = extra
            self._emit(payload)
        except Exception:
            logger.debug("[LLMTruthDiff] log_l3_model_io failed", exc_info=True)


_default_logger = LLMTruthDiffLogger()


def get_llm_truth_diff_logger() -> LLMTruthDiffLogger:
    return _default_logger
