"""
CPU-bound Qwen2-0.5B sidecar — serial task queue for async cognition.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from core.sidecar_llm import DEFAULT_MODEL_REL_PATH, default_sidecar_model_path
from core.sidecar_telemetry import get_sidecar_telemetry
from core.sidecar_prompts import (
    CHATML_STOPS,
    build_prompt_for_task,
    parse_task_output,
    task_inference_params,
)
from core.sidecar_types import SidecarResult, SidecarTask

logger = logging.getLogger("Qube.SidecarLLMWorker")

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore


class SidecarLlmWorker(QThread):
    """Owns a single Llama instance; all inference runs on this thread."""

    title_generated = pyqtSignal(str, str)
    ingest_blurb_ready = pyqtSignal(str, str)  # filename, blurb
    sidecar_telemetry_updated = pyqtSignal(dict)

    def __init__(self, db_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.db = db_manager
        self._cmd_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.model = None
        self.model_loaded = False
        self._warned_missing = False
        self.telemetry = get_sidecar_telemetry()

    def enqueue_task(
        self,
        task: SidecarTask,
        payload: dict,
        out: list,
        done_event: threading.Event,
    ) -> None:
        self._cmd_queue.put(
            {
                "op": "task",
                "task": task,
                "payload": payload,
                "out": out,
                "done_event": done_event,
            }
        )

    def enqueue_raw_prompt(
        self,
        prompt: str,
        out: list,
        done_event: threading.Event,
        *,
        timeout_hint: float = 120.0,
    ) -> None:
        self._cmd_queue.put(
            {
                "op": "raw",
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.2,
                "out": out,
                "done_event": done_event,
            }
        )

    def enqueue_title(self, user_prompt: str, session_id: str) -> None:
        self._cmd_queue.put(
            {
                "op": "title",
                "user_prompt": user_prompt,
                "session_id": session_id,
            }
        )

    def enqueue_ingest_blurb(
        self, filename: str, sample_text: str, out: list | None = None
    ) -> None:
        self._cmd_queue.put(
            {
                "op": "ingest_blurb",
                "filename": filename,
                "sample_text": sample_text,
                "out": out,
            }
        )

    def stop_engine(self) -> None:
        self._stop.set()
        self._cmd_queue.put({"op": "shutdown"})

    def _sync_telemetry_runtime(self, *, degraded_reason: str = "") -> None:
        self.telemetry.set_runtime_state(
            model_loaded=bool(self.model_loaded),
            degraded_reason=degraded_reason,
        )
        try:
            self.telemetry.set_queue_depth(self._cmd_queue.qsize())
        except Exception:
            self.telemetry.set_queue_depth(0)
        try:
            self.sidecar_telemetry_updated.emit(self.telemetry.summarize())
        except Exception as e:
            logger.debug("[Sidecar] telemetry emit skipped: %s", e)

    def run(self) -> None:
        if Llama is None:
            logger.error("[Sidecar] llama_cpp not available")
            self._sync_telemetry_runtime(degraded_reason="llama_cpp_unavailable")
            self._run_degraded_queue_loop()
            return

        path = default_sidecar_model_path()
        if not os.path.isfile(path):
            if not self._warned_missing:
                logger.warning("[Sidecar] Model not found at %s — sidecar disabled", path)
                self._warned_missing = True
            self._sync_telemetry_runtime(degraded_reason="model_not_found")
            self._run_degraded_queue_loop()
            return

        try:
            logger.info("[Sidecar] Loading Qwen2-0.5B on CPU (%s)", path)
            self.model = Llama(
                model_path=path,
                n_gpu_layers=0,
                n_ctx=2048,
                verbose=False,
            )
            self.model_loaded = True
            self._sync_telemetry_runtime()
        except Exception as e:
            logger.error("[Sidecar] Load failed: %s", e)
            self._sync_telemetry_runtime(degraded_reason=f"load_failed:{e}")
            self._run_degraded_queue_loop()
            return

        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                self.telemetry.set_queue_depth(self._cmd_queue.qsize())
                continue

            self.telemetry.set_queue_depth(self._cmd_queue.qsize())
            op = cmd.get("op")
            if op == "shutdown":
                break
            try:
                if op == "title":
                    self._do_title(cmd)
                elif op == "task":
                    self._do_task(cmd)
                elif op == "raw":
                    self._do_raw(cmd)
                elif op == "ingest_blurb":
                    self._do_ingest_blurb(cmd)
            except Exception as e:
                logger.exception("[Sidecar] command failed op=%s: %s", op, e)
                if op == "task":
                    out = cmd.get("out")
                    if isinstance(out, list):
                        out.append(
                            SidecarResult(
                                ok=False,
                                error=str(e),
                                task=cmd.get("task"),
                            )
                        )
                    ev = cmd.get("done_event")
                    if ev is not None:
                        ev.set()
                elif op == "raw":
                    out = cmd.get("out")
                    if isinstance(out, list):
                        out.append("")
                    ev = cmd.get("done_event")
                    if ev is not None:
                        ev.set()

        self.model = None
        self.model_loaded = False

    def _run_degraded_queue_loop(self) -> None:
        """Drain queue with failures so waiters never block forever."""
        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if cmd.get("op") == "shutdown":
                break
            self._fail_command(cmd, "model_unavailable")

    def _fail_command(self, cmd: dict, reason: str) -> None:
        op = cmd.get("op")
        task = cmd.get("task")
        if op == "task" and task is not None:
            self.telemetry.record(
                task,
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason=reason,
            )
        elif op == "title":
            self.telemetry.record(
                SidecarTask.title,
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason=reason,
            )
        elif op == "ingest_blurb":
            self.telemetry.record(
                SidecarTask.ingest_blurb,
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason=reason,
            )
        if op == "task":
            out = cmd.get("out")
            if isinstance(out, list):
                out.append(
                    SidecarResult(
                        ok=False,
                        error=reason,
                        task=cmd.get("task"),
                    )
                )
            ev = cmd.get("done_event")
            if ev is not None:
                ev.set()
        elif op == "raw":
            out = cmd.get("out")
            if isinstance(out, list):
                out.append("")
            ev = cmd.get("done_event")
            if ev is not None:
                ev.set()
        elif op == "ingest_blurb":
            out = cmd.get("out")
            if isinstance(out, list):
                out.append("")

    def _complete_prompt(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> str:
        if not self.model:
            return ""
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=CHATML_STOPS,
            )
            return (output.get("choices") or [{}])[0].get("text") or ""
        except Exception as e:
            logger.debug("[Sidecar] inference error: %s", e)
            return ""

    def _do_task(self, cmd: dict) -> None:
        task: SidecarTask = cmd["task"]
        payload = cmd.get("payload") or {}
        params = task_inference_params(task)
        prompt = build_prompt_for_task(task, **payload)
        raw = self._complete_prompt(
            prompt,
            max_tokens=int(params.get("max_tokens", 128)),
            temperature=float(params.get("temperature", 0.2)),
        )
        result = parse_task_output(task, raw, **payload)
        out = cmd.get("out")
        if isinstance(out, list):
            out.append(result)
        ev = cmd.get("done_event")
        if ev is not None:
            ev.set()
        self._sync_telemetry_runtime()

    def _do_raw(self, cmd: dict) -> None:
        raw = self._complete_prompt(
            cmd.get("prompt") or "",
            max_tokens=int(cmd.get("max_tokens", 256)),
            temperature=float(cmd.get("temperature", 0.2)),
        )
        out = cmd.get("out")
        if isinstance(out, list):
            out.append(raw)
        ev = cmd.get("done_event")
        if ev is not None:
            ev.set()

    def _do_title(self, cmd: dict) -> None:
        session_id = str(cmd.get("session_id") or "")
        user_prompt = cmd.get("user_prompt") or ""
        t0 = time.perf_counter()
        result = parse_task_output(
            SidecarTask.title,
            self._complete_prompt(
                build_prompt_for_task(SidecarTask.title, user_prompt=user_prompt),
                max_tokens=12,
                temperature=0.2,
            ),
        )
        new_title = (result.parsed or {}).get("title") or result.text
        ok = bool(new_title)
        if new_title and self.db and session_id:
            if self.db.rename_session(session_id, new_title):
                self.title_generated.emit(session_id, new_title)
        self.telemetry.record(
            SidecarTask.title,
            ok=ok,
            latency_ms=(time.perf_counter() - t0) * 1000,
            foreground=False,
            reason="" if ok else "empty",
        )
        self._sync_telemetry_runtime()

    def _do_ingest_blurb(self, cmd: dict) -> None:
        filename = str(cmd.get("filename") or "")
        sample = cmd.get("sample_text") or ""
        t0 = time.perf_counter()
        result = parse_task_output(
            SidecarTask.ingest_blurb,
            self._complete_prompt(
                build_prompt_for_task(SidecarTask.ingest_blurb, sample_text=sample),
                max_tokens=48,
                temperature=0.2,
            ),
        )
        blurb = (result.parsed or {}).get("blurb") or result.text
        ok = bool(blurb)
        if blurb and filename:
            self.ingest_blurb_ready.emit(filename, blurb)
        out = cmd.get("out")
        if isinstance(out, list):
            out.append(blurb)
        self.telemetry.record(
            SidecarTask.ingest_blurb,
            ok=ok,
            latency_ms=(time.perf_counter() - t0) * 1000,
            foreground=False,
            reason="" if ok else "empty",
        )
        self._sync_telemetry_runtime()
