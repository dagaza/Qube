"""
Provider-agnostic canonical LLM request normalization for debugging exports.

Enable trace logging with ENABLE_CANONICAL_TRACE_EXPORT=1 (logs to Qube.NativeLLM.Debug).

Optional format adapters live in core.canonical_request_adapters (serialization only).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("Qube.NativeLLM.Debug")

_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})

# Internal keys consumed when building the canonical shape (remainder -> metadata).
_CANONICAL_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "model_name",
        "model_id",
        "messages",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "stop_tokens",
        "stops",
        "prompt",
    }
)

_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    }
)


def canonical_trace_export_enabled() -> bool:
    return os.environ.get("ENABLE_CANONICAL_TRACE_EXPORT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class CanonicalSampling:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


@dataclass
class CanonicalMessage:
    role: str
    content: str


@dataclass
class CanonicalRequest:
    model: str
    messages: list[CanonicalMessage]
    sampling: CanonicalSampling
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in self.messages
            ],
            "sampling": asdict(self.sampling),
            "stop": list(self.stop),
            "metadata": dict(self.metadata),
        }


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        iv = int(value)
        return iv if iv > 0 else None
    except (TypeError, ValueError):
        return None


def _normalize_role(raw_role: Any) -> tuple[str, dict[str, Any]]:
    role = str(raw_role or "user").strip().lower()
    if role in _ALLOWED_ROLES:
        return role, {}
    return "user", {"original_role": role}


def _coerce_messages(raw: Any) -> tuple[list[CanonicalMessage], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], []
    messages: list[CanonicalMessage] = []
    role_notes: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        role, note = _normalize_role(item.get("role"))
        content = str(item.get("content") or "")
        messages.append(CanonicalMessage(role=role, content=content))
        if note:
            role_notes.append({"index": idx, **note})
    return messages, role_notes


def _extract_stop(raw: dict[str, Any]) -> list[str]:
    for key in ("stop", "stop_tokens", "stops"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            s = value.strip()
            return [s] if s else []
        if isinstance(value, list):
            out = [str(v).strip() for v in value if str(v).strip()]
            if out:
                return out
    return []


def _extract_model(raw: dict[str, Any]) -> str:
    for key in ("model", "model_name", "model_id"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _build_metadata(
    raw: dict[str, Any],
    *,
    role_notes: list[dict[str, Any]],
    prompt_only: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _CANONICAL_TOP_LEVEL_KEYS:
            continue
        if key in _SAMPLING_KEYS:
            continue
        metadata[key] = value
    if role_notes:
        metadata["role_normalization"] = role_notes
    if prompt_only:
        metadata["input_mode"] = "completion_prompt"
        metadata["prompt"] = prompt_only
    return metadata


class CanonicalRequestExporter:
    """Normalize heterogeneous internal engine payloads into CanonicalRequest."""

    @staticmethod
    def export_canonical_request(internal_request: dict) -> CanonicalRequest:
        raw = dict(internal_request or {})
        messages, role_notes = _coerce_messages(raw.get("messages"))
        prompt_only = ""
        if not messages:
            prompt_val = raw.get("prompt")
            if prompt_val is not None and str(prompt_val).strip():
                prompt_only = str(prompt_val)

        sampling = CanonicalSampling(
            temperature=_as_float(raw.get("temperature"), 1.0),
            top_p=_as_float(raw.get("top_p"), 1.0),
            top_k=_as_optional_int(raw.get("top_k")),
            repeat_penalty=_as_optional_float(raw.get("repeat_penalty")),
            presence_penalty=_as_optional_float(raw.get("presence_penalty")),
            frequency_penalty=_as_optional_float(raw.get("frequency_penalty")),
        )
        metadata = _build_metadata(
            raw,
            role_notes=role_notes,
            prompt_only=prompt_only,
        )
        return CanonicalRequest(
            model=_extract_model(raw),
            messages=messages,
            sampling=sampling,
            stop=_extract_stop(raw),
            metadata=metadata,
        )


def build_canonical_request_trace_payload(
    internal_request: dict,
    *,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    canonical = CanonicalRequestExporter.export_canonical_request(internal_request)
    payload: dict[str, Any] = {
        "event": "canonical_request_trace",
        "timestamp": _utc_timestamp(),
        "canonical": canonical.to_dict(),
    }
    if context:
        for key in ("request_id", "exchange_id", "session_id", "model_name", "engine_mode"):
            if key in context and context[key] is not None:
                payload[key] = context[key]
    return payload


def log_canonical_request_trace(
    internal_request: dict,
    *,
    context: Optional[dict[str, Any]] = None,
) -> None:
    if not canonical_trace_export_enabled():
        return
    try:
        payload = build_canonical_request_trace_payload(
            internal_request,
            context=context,
        )
        wrapped = {"canonical_request_trace": payload}
        logger.info(json.dumps(wrapped, ensure_ascii=False))
        logger.info(
            "[CanonicalRequestTrace] model=%r messages=%d stops=%d",
            payload.get("canonical", {}).get("model") or "(unset)",
            len(payload.get("canonical", {}).get("messages") or []),
            len(payload.get("canonical", {}).get("stop") or []),
        )
    except Exception:
        logger.debug("[CanonicalRequestTrace] export failed", exc_info=True)
