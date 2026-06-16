"""
Sidecar LLM contracts — CPU-bound Qwen3 1.7B for structured async cognition.

Pure types + client facade; inference runs on ``workers.sidecar_llm_worker``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from core.sidecar_telemetry import get_sidecar_telemetry
from core.sidecar_types import QueryExpansion, SidecarResult, SidecarTask

_FOREGROUND_TASKS = frozenset(
    {SidecarTask.query_rewrite, SidecarTask.source_digest}
)

__all__ = [
    "SidecarTask",
    "SidecarResult",
    "QueryExpansion",
    "SidecarLlmClient",
    "default_sidecar_model_path",
    "sidecar_model_available",
]

logger = logging.getLogger("Qube.SidecarLLM")

from core.auxiliary_cognition import (
    BUNDLED_DEFAULT_REL_PATH as DEFAULT_MODEL_REL_PATH,
    bundled_default_path,
    cognition_model_available,
    resolve_active_cognition_path,
)


def default_sidecar_model_path() -> str:
    """Backward-compatible alias for bundled default path."""
    return bundled_default_path()


def sidecar_model_available() -> bool:
    return cognition_model_available()


def active_sidecar_model_path() -> str:
    return resolve_active_cognition_path()


class SidecarLlmClient:
    """
    Thread-safe facade for background/foreground sidecar jobs.

    Implements the ``generate`` / ``isRunning`` surface ``EnrichmentWorker`` expects
    for cognition paths (``isRunning`` is always False — no chat contention).
    """

    def __init__(self, worker: Any) -> None:
        self._worker = worker

    @property
    def available(self) -> bool:
        return bool(getattr(self._worker, "model_loaded", False))

    def isRunning(self) -> bool:
        return False

    def complete(
        self,
        task: SidecarTask,
        *,
        timeout_sec: float = 120.0,
        **kwargs: Any,
    ) -> SidecarResult:
        tel = get_sidecar_telemetry()
        t0 = time.perf_counter()
        foreground = task in _FOREGROUND_TASKS
        if self._worker is None:
            result = SidecarResult(ok=False, error="no_worker", task=task)
            tel.record(
                task,
                ok=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                foreground=foreground,
                reason="no_worker",
            )
            return result
        out: list[SidecarResult] = []
        ev = threading.Event()
        self._worker.enqueue_task(task, kwargs, out, ev)
        if not ev.wait(timeout_sec):
            tel.record(
                task,
                ok=False,
                latency_ms=max(0.0, (time.perf_counter() - t0) * 1000),
                wait_ms=0.0,
                foreground=foreground,
                reason="timeout",
            )
            return SidecarResult(ok=False, error="timeout", task=task)
        result = out[0] if out else SidecarResult(ok=False, error="empty", task=task)
        wait_ms = float(getattr(result, "queue_wait_ms", 0.0) or 0.0)
        inference_ms = float(getattr(result, "inference_ms", 0.0) or 0.0)
        if inference_ms <= 0.0:
            inference_ms = max(0.0, (time.perf_counter() - t0) * 1000 - wait_ms)
        tel.record(
            task,
            ok=bool(result.ok),
            latency_ms=inference_ms,
            wait_ms=wait_ms,
            foreground=foreground,
            reason="" if result.ok else (result.error or "fail"),
        )
        return result

    def generate(self, prompt: str, *, timeout_sec: float = 120.0) -> str:
        """Legacy raw-prompt path (episode-sized prompts built upstream)."""
        if self._worker is None:
            get_sidecar_telemetry().record(
                "raw_prompt",
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason="no_worker",
            )
            return ""
        out: list[str] = []
        ev = threading.Event()
        self._worker.enqueue_raw_prompt(prompt, out, ev, timeout_hint=timeout_sec)
        if not ev.wait(timeout_sec):
            get_sidecar_telemetry().record(
                "raw_prompt",
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason="timeout",
            )
            return ""
        return (out[0] if out else "") or ""

    def enqueue_title(
        self,
        user_prompt: str,
        session_id: str,
        *,
        assistant_reply: str = "",
    ) -> None:
        if self._worker is not None:
            self._worker.enqueue_title(
                user_prompt,
                session_id,
                assistant_reply=assistant_reply,
            )

    def request_companion_line(self, payload: dict) -> bool:
        """Fire-and-forget companion caption generation (queued on the sidecar thread)."""
        if self._worker is None:
            return False
        enqueue = getattr(self._worker, "enqueue_companion_line", None)
        if not callable(enqueue):
            return False
        return bool(enqueue(dict(payload or {})))

    def preview_companion_line(self, **kwargs: Any) -> SidecarResult:
        """Blocking preview for Settings test (bypasses queue-drop policy)."""
        return self.complete(
            SidecarTask.companion_line,
            timeout_sec=90.0,
            **kwargs,
        )


def merge_sidecar_enabled_setting() -> bool:
    from core.app_settings import get_sidecar_enabled

    return get_sidecar_enabled()
