"""
Sidecar LLM contracts — CPU-bound Qwen2-0.5B for structured async cognition.

Pure types + client facade; inference runs on ``workers.sidecar_llm_worker``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from core.sidecar_types import QueryExpansion, SidecarResult, SidecarTask

__all__ = [
    "SidecarTask",
    "SidecarResult",
    "QueryExpansion",
    "SidecarLlmClient",
    "default_sidecar_model_path",
    "sidecar_model_available",
]

logger = logging.getLogger("Qube.SidecarLLM")

DEFAULT_MODEL_REL_PATH = os.path.join("models", "qwen2-0_5b-instruct-q4_k_m.gguf")


def default_sidecar_model_path() -> str:
    return os.path.join(os.getcwd(), DEFAULT_MODEL_REL_PATH)


def sidecar_model_available() -> bool:
    return os.path.isfile(default_sidecar_model_path())


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
        if self._worker is None:
            return SidecarResult(ok=False, error="no_worker", task=task)
        out: list[SidecarResult] = []
        ev = threading.Event()
        self._worker.enqueue_task(task, kwargs, out, ev)
        if not ev.wait(timeout_sec):
            return SidecarResult(ok=False, error="timeout", task=task)
        return out[0] if out else SidecarResult(ok=False, error="empty", task=task)

    def generate(self, prompt: str, *, timeout_sec: float = 120.0) -> str:
        """Legacy raw-prompt path (episode-sized prompts built upstream)."""
        if self._worker is None:
            return ""
        out: list[str] = []
        ev = threading.Event()
        self._worker.enqueue_raw_prompt(prompt, out, ev, timeout_hint=timeout_sec)
        if not ev.wait(timeout_sec):
            return ""
        return (out[0] if out else "") or ""

    def enqueue_title(self, user_prompt: str, session_id: str) -> None:
        if self._worker is not None:
            self._worker.enqueue_title(user_prompt, session_id)


def merge_sidecar_enabled_setting() -> bool:
    from core.app_settings import get_sidecar_enabled

    return get_sidecar_enabled()
