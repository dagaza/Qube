"""
In-memory telemetry for the CPU sidecar (Qwen3 1.7B assistive cognition).

Observer-only aggregates for health, latency, and effectiveness proxies.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from core.app_settings import get_sidecar_enabled
from core.sidecar_types import SidecarTask

logger = logging.getLogger("Qube.Sidecar.Telemetry")

_SIDECAR_DEBUG_ENV = "QUBE_SIDECAR_DEBUG"
_MAX_EVENTS = 400
_MAX_TURN_EVENTS = 200


@dataclass
class SidecarRuntimeState:
    enabled: bool = False
    model_on_disk: bool = False
    model_loaded: bool = False
    degraded_reason: str = ""
    active_model_basename: str = ""
    is_bundled_default: bool = True


class SidecarTelemetryBrain:
    """Thread-safe ring buffer + rolling counters for sidecar observability."""

    def __init__(self, max_events: int = _MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict] = deque(maxlen=max_events)
        self._turn_events: deque[dict] = deque(maxlen=_MAX_TURN_EVENTS)
        self._runtime = SidecarRuntimeState()
        self._queue_depth = 0

    def set_runtime_state(
        self,
        *,
        model_loaded: bool | None = None,
        degraded_reason: str | None = None,
        active_model_basename: str | None = None,
        is_bundled_default: bool | None = None,
    ) -> None:
        with self._lock:
            if model_loaded is not None:
                self._runtime.model_loaded = bool(model_loaded)
            if degraded_reason is not None:
                self._runtime.degraded_reason = str(degraded_reason or "")
            if active_model_basename is not None:
                self._runtime.active_model_basename = str(active_model_basename or "")
            if is_bundled_default is not None:
                self._runtime.is_bundled_default = bool(is_bundled_default)
            self._runtime.enabled = get_sidecar_enabled()
            from core.auxiliary_cognition import cognition_model_available

            self._runtime.model_on_disk = cognition_model_available()

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = max(0, int(depth))

    def record(
        self,
        task: str | SidecarTask,
        *,
        ok: bool,
        latency_ms: float = 0.0,
        foreground: bool = False,
        reason: str | None = None,
        wait_ms: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        task_name = task.value if isinstance(task, SidecarTask) else str(task)
        event = {
            "ts": time.time(),
            "task": task_name,
            "ok": bool(ok),
            "latency_ms": round(float(latency_ms), 2),
            "wait_ms": round(float(wait_ms), 2),
            "foreground": bool(foreground),
            "reason": reason or "",
            "meta": dict(meta or {}),
        }
        with self._lock:
            self._events.append(event)
        if _debug_enabled():
            logger.info("[SidecarTelemetry] %s", event)
        else:
            logger.debug("[SidecarTelemetry] task=%s ok=%s reason=%s", task_name, ok, reason or "")

    def record_turn(
        self,
        *,
        rewrite_attempted: bool = False,
        rewrite_applied: bool = False,
        rewrite_reason: str = "",
        rewrite_confidence: float = 0.0,
        digest_memory_attempted: bool = False,
        digest_memory_applied: bool = False,
        digest_rag_attempted: bool = False,
        digest_rag_applied: bool = False,
        foreground_sidecar_ms: float = 0.0,
        hybrid_extra_memory: int = 0,
        hybrid_extra_rag: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "ts": time.time(),
            "rewrite_attempted": rewrite_attempted,
            "rewrite_applied": rewrite_applied,
            "rewrite_reason": rewrite_reason,
            "rewrite_confidence": round(float(rewrite_confidence), 3),
            "digest_memory_attempted": digest_memory_attempted,
            "digest_memory_applied": digest_memory_applied,
            "digest_rag_attempted": digest_rag_attempted,
            "digest_rag_applied": digest_rag_applied,
            "foreground_sidecar_ms": round(float(foreground_sidecar_ms), 2),
            "hybrid_extra_memory": int(hybrid_extra_memory),
            "hybrid_extra_rag": int(hybrid_extra_rag),
            "meta": dict(meta or {}),
        }
        with self._lock:
            self._turn_events.append(event)

    def summarize(self) -> dict[str, Any]:
        from core.auxiliary_cognition import (
            active_cognition_basename,
            is_active_cognition_bundled,
        )

        self.set_runtime_state(
            active_model_basename=active_cognition_basename(),
            is_bundled_default=is_active_cognition_bundled(),
        )
        with self._lock:
            events = list(self._events)
            turns = list(self._turn_events)
            runtime = SidecarRuntimeState(
                enabled=self._runtime.enabled,
                model_on_disk=self._runtime.model_on_disk,
                model_loaded=self._runtime.model_loaded,
                degraded_reason=self._runtime.degraded_reason,
                active_model_basename=self._runtime.active_model_basename,
                is_bundled_default=self._runtime.is_bundled_default,
            )
            queue_depth = self._queue_depth

        return _build_summary(events, turns, runtime, queue_depth)

    def get_summary(self) -> dict[str, Any]:
        return self.summarize()

    def companion_line_stats(self, *, window: int = 20) -> dict[str, Any]:
        """Rolling ok-rate for companion_line task (expression capability downgrade)."""
        with self._lock:
            events = [
                e
                for e in self._events
                if str(e.get("task") or "") == SidecarTask.companion_line.value
            ][-window:]
        total = len(events)
        if not total:
            return {"total": 0, "ok_rate": 1.0, "low_quality": 0}
        ok = sum(1 for e in events if e.get("ok"))
        low_q = sum(
            1
            for e in events
            if str(e.get("reason") or "") in ("low_quality", "parse_fail", "skip")
        )
        return {
            "total": total,
            "ok_rate": ok / total,
            "low_quality": low_q,
        }


def _debug_enabled() -> bool:
    return os.environ.get(_SIDECAR_DEBUG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _build_summary(
    events: list[dict],
    turns: list[dict],
    runtime: SidecarRuntimeState,
    queue_depth: int,
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for e in events:
        t = e.get("task") or "unknown"
        bucket = by_task.setdefault(
            t,
            {
                "attempts": 0,
                "ok": 0,
                "fail": 0,
                "timeout": 0,
                "foreground_attempts": 0,
                "foreground_timeout": 0,
                "latencies": [],
                "reasons": {},
            },
        )
        bucket["attempts"] += 1
        if e.get("ok"):
            bucket["ok"] += 1
        else:
            bucket["fail"] += 1
        reason = str(e.get("reason") or "")
        if reason == "timeout":
            bucket["timeout"] += 1
        if reason:
            bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
        if e.get("foreground"):
            bucket["foreground_attempts"] += 1
            if reason == "timeout":
                bucket["foreground_timeout"] += 1
        bucket["latencies"].append(float(e.get("latency_ms") or 0))

    fg_latencies = [
        float(e.get("latency_ms") or 0)
        for e in events
        if e.get("foreground")
    ]
    bg_latencies = [
        float(e.get("latency_ms") or 0)
        for e in events
        if not e.get("foreground")
    ]

    rewrite_attempts = sum(1 for t in turns if t.get("rewrite_attempted"))
    rewrite_applied = sum(1 for t in turns if t.get("rewrite_applied"))
    digest_mem_attempts = sum(1 for t in turns if t.get("digest_memory_attempted"))
    digest_mem_applied = sum(1 for t in turns if t.get("digest_memory_applied"))
    digest_rag_attempts = sum(1 for t in turns if t.get("digest_rag_attempted"))
    digest_rag_applied = sum(1 for t in turns if t.get("digest_rag_applied"))
    hybrid_mem_extra = sum(int(t.get("hybrid_extra_memory") or 0) for t in turns)
    hybrid_rag_extra = sum(int(t.get("hybrid_extra_rag") or 0) for t in turns)
    fg_turn_ms = [float(t.get("foreground_sidecar_ms") or 0) for t in turns]

    total_attempts = len(events)
    total_ok = sum(1 for e in events if e.get("ok"))

    health, health_tip = _health_status(
        runtime,
        queue_depth,
        total_attempts,
        total_ok,
        fg_latencies,
        rewrite_attempts,
        rewrite_applied,
    )

    return {
        "runtime": {
            "enabled": runtime.enabled,
            "model_on_disk": runtime.model_on_disk,
            "model_loaded": runtime.model_loaded,
            "degraded_reason": runtime.degraded_reason,
            "active_model_basename": runtime.active_model_basename,
            "is_bundled_default": runtime.is_bundled_default,
            "status": _status_label(runtime, queue_depth),
        },
        "queue_depth": queue_depth,
        "total_invocations": total_attempts,
        "success_rate": (total_ok / total_attempts) if total_attempts else 0.0,
        "by_task": {
            k: {
                "attempts": v["attempts"],
                "ok": v["ok"],
                "fail": v["fail"],
                "timeout": v["timeout"],
                "avg_latency_ms": (
                    sum(v["latencies"]) / len(v["latencies"]) if v["latencies"] else 0.0
                ),
                "p95_latency_ms": _percentile(v["latencies"], 95),
                "top_reasons": sorted(
                    v["reasons"].items(), key=lambda x: -x[1]
                )[:3],
            }
            for k, v in by_task.items()
        },
        "foreground": {
            "attempts": len(fg_latencies),
            "timeout_rate": (
                sum(1 for e in events if e.get("foreground") and e.get("reason") == "timeout")
                / len(fg_latencies)
                if fg_latencies
                else 0.0
            ),
            "p95_latency_ms": _percentile(fg_latencies, 95),
            "p95_turn_ms": _percentile(fg_turn_ms, 95),
        },
        "background": {
            "attempts": len(bg_latencies),
            "p95_latency_ms": _percentile(bg_latencies, 95),
        },
        "turns_sampled": len(turns),
        "rewrite": {
            "attempted": rewrite_attempts,
            "applied": rewrite_applied,
            "apply_rate": (
                rewrite_applied / rewrite_attempts if rewrite_attempts else 0.0
            ),
        },
        "digest": {
            "memory_attempted": digest_mem_attempts,
            "memory_applied": digest_mem_applied,
            "rag_attempted": digest_rag_attempts,
            "rag_applied": digest_rag_applied,
        },
        "hybrid": {
            "extra_memory_hits": hybrid_mem_extra,
            "extra_rag_hits": hybrid_rag_extra,
        },
        "health": health,
        "health_tip": health_tip,
    }


def _status_label(runtime: SidecarRuntimeState, queue_depth: int) -> str:
    if not runtime.enabled:
        return "Off"
    if not runtime.model_on_disk:
        return "No model"
    if runtime.degraded_reason:
        return "Degraded"
    if runtime.model_loaded:
        if queue_depth > 8:
            return "Busy"
        return "Online"
    return "Starting"


def _health_status(
    runtime: SidecarRuntimeState,
    queue_depth: int,
    total_attempts: int,
    total_ok: int,
    fg_latencies: list[float],
    rewrite_attempts: int,
    rewrite_applied: int,
) -> tuple[str, str]:
    if not runtime.enabled:
        return "⚪ Disabled", "Sidecar disabled in settings or GGUF missing."
    if not runtime.model_on_disk:
        return "⚪ No model", "Place Qwen3-1.7B-Q6_K.gguf under models/cognition/."
    if runtime.degraded_reason:
        return "🔴 Degraded", runtime.degraded_reason
    if not runtime.model_loaded:
        return "🟡 Starting", "Sidecar worker has not finished loading the model."

    tips: list[str] = []
    if queue_depth > 12:
        tips.append(f"Queue depth high ({queue_depth}).")
    if total_attempts:
        fail_rate = 1.0 - (total_ok / total_attempts)
        if fail_rate > 0.25:
            tips.append(f"Failure rate {fail_rate:.0%} over last {total_attempts} calls.")
    fg_timeouts = sum(1 for _ in fg_latencies)  # placeholder — recalc in summarize if needed
    if fg_latencies and _percentile(fg_latencies, 95) > 1200:
        tips.append("Foreground sidecar p95 > 1.2s — consider raising timeout or disabling digest.")

    if tips:
        return "🟡 Watch", " ".join(tips)
    if rewrite_attempts and rewrite_applied == 0:
        return "🟡 Idle rewrite", "Follow-up rewrites attempted but none applied (confidence/guards)."
    return "🟢 Healthy", "Sidecar online with normal failure and latency profile."


_brain: SidecarTelemetryBrain | None = None
_brain_lock = threading.Lock()


def get_sidecar_telemetry() -> SidecarTelemetryBrain:
    global _brain
    with _brain_lock:
        if _brain is None:
            _brain = SidecarTelemetryBrain()
        return _brain


def get_sidecar_telemetry_brain() -> SidecarTelemetryBrain:
    """Alias for companion cognition capability routing."""
    return get_sidecar_telemetry()
