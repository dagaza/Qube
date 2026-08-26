"""
CPU-bound auxiliary cognition — priority task queue for async sidecar inference.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from core.auxiliary_cognition import (
    cognition_n_ctx_for_path,
    resolve_active_cognition_path,
)
from core.cognition_prompt_adapter import (
    apply_qwen3_no_think_to_prompt,
    cognition_stop_tokens,
    resolve_cognition_chat_format,
)
from core.sidecar_engine_queue import (
    INGEST_BLURB_MAX_QUEUED,
    SidecarCommandQueue,
    should_defer_companion_line,
    should_drop_ingest_blurb,
)
from core.app_settings import (
    get_sidecar_title_context_mode,
    get_sidecar_title_inference_profile,
)
from core.title_generation_experiment import log_title_experiment_run, run_title_generation
from core.title_inference_profiles import get_title_profile
from core.sidecar_prompts import normalize_title_user_prompt
from core.sidecar_telemetry import get_sidecar_telemetry
from core.inference_transparency import log_inference_transparency, snapshot_from_loaded_llama
from core.sidecar_prompts import (
    build_prompt_for_task,
    parse_task_output,
    task_inference_params,
)
from core.sidecar_types import SidecarResult, SidecarTask

from core.llama_cpp_import import get_llama_class

logger = logging.getLogger("Qube.SidecarLLMWorker")

try:
    import queue as _queue_mod
except ImportError:
    _queue_mod = None  # type: ignore


def _queue_wait_ms(cmd: dict) -> float:
    submitted = cmd.get("submitted_at")
    dequeued = cmd.get("dequeued_at")
    if submitted is None or dequeued is None:
        return 0.0
    return max(0.0, (float(dequeued) - float(submitted)) * 1000.0)


class SidecarLlmWorker(QThread):
    """Owns a single Llama instance; all inference runs on this thread."""

    title_generated = pyqtSignal(str, str)
    ingest_blurb_ready = pyqtSignal(str, str)  # filename, blurb
    companion_line_ready = pyqtSignal(str, str, str)  # line, kind, trigger
    sidecar_telemetry_updated = pyqtSignal(dict)
    model_reload_finished = pyqtSignal(bool, str)

    def __init__(self, db_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.db = db_manager
        self._cmd_queue = SidecarCommandQueue()
        self._stop = threading.Event()
        self._reloading = False
        self.model = None
        self.model_loaded = False
        self.active_model_path: str = ""
        self.active_chat_format: str = "chatml"
        self._warned_missing = False
        self.telemetry = get_sidecar_telemetry()
        self._inference_transparency: dict[str, Any] = {}

    def get_inference_transparency(self) -> dict[str, Any]:
        """Read-only sidecar stack snapshot for UI telemetry."""
        snap = dict(self._inference_transparency)
        snap.setdefault("role", "sidecar")
        snap.setdefault("backend", "cpu")
        snap.setdefault("compute_mode", "cpu")
        snap["loaded"] = bool(self.model_loaded)
        if not self.model_loaded:
            try:
                degraded = str(self.telemetry.summarize().get("degraded_reason") or "")
            except Exception:
                degraded = ""
            if degraded:
                snap["degraded_reason"] = degraded
        return snap

    def reload_from_settings(self) -> None:
        """Enqueue hot-reload of the cognition model from current settings."""
        self._put_cmd({"op": "reload"})

    def enqueue_task(
        self,
        task: SidecarTask,
        payload: dict,
        out: list,
        done_event: threading.Event,
    ) -> None:
        self._put_cmd(
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
        self._put_cmd(
            {
                "op": "raw",
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.2,
                "out": out,
                "done_event": done_event,
            }
        )

    def enqueue_title(
        self,
        user_prompt: str,
        session_id: str,
        *,
        assistant_reply: str = "",
    ) -> None:
        self._put_cmd(
            {
                "op": "title",
                "user_prompt": user_prompt,
                "assistant_reply": assistant_reply,
                "session_id": session_id,
            }
        )

    def enqueue_ingest_blurb(
        self, filename: str, sample_text: str, out: list | None = None
    ) -> bool:
        """Queue ingest blurb; coalesces per filename and caps pending blurbs."""
        filename = str(filename or "")
        if not filename:
            return False

        removed = self._cmd_queue.purge(
            lambda c: c.get("op") == "ingest_blurb" and c.get("filename") == filename
        )
        if removed:
            self.telemetry.record(
                SidecarTask.ingest_blurb,
                ok=True,
                latency_ms=0.0,
                foreground=False,
                reason="coalesced",
                meta={"replaced_pending": removed, "filename": filename},
            )

        if should_drop_ingest_blurb(
            self._cmd_queue.count(lambda c: c.get("op") == "ingest_blurb")
        ):
            self.telemetry.record(
                SidecarTask.ingest_blurb,
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason="ingest_queue_saturated",
                meta={"filename": filename, "max_queued": INGEST_BLURB_MAX_QUEUED},
            )
            logger.info(
                "[Sidecar] ingest blurb dropped (queue saturated) file=%s",
                filename,
            )
            return False

        self._put_cmd(
            {
                "op": "ingest_blurb",
                "filename": filename,
                "sample_text": sample_text,
                "out": out,
            }
        )
        return True

    def enqueue_companion_line(self, payload: dict) -> bool:
        """Fire-and-forget companion caption; deferred when queue is deep."""
        if self._reloading or not self.model_loaded:
            return False
        depth = self._cmd_queue.qsize()
        if should_defer_companion_line(depth):
            trigger = str((payload or {}).get("trigger") or "idle")
            self.telemetry.record(
                SidecarTask.companion_line,
                ok=False,
                latency_ms=0.0,
                foreground=False,
                reason="queue_deferred",
                meta={"queue_depth": depth, "trigger": trigger},
            )
            logger.info(
                "[Sidecar] companion line deferred trigger=%s depth=%d",
                trigger,
                depth,
            )
            return False
        self._put_cmd({"op": "companion_line", "payload": dict(payload or {})})
        return True

    def stop_engine(self) -> None:
        self._stop.set()
        self._put_cmd({"op": "shutdown"})

    def _put_cmd(self, cmd: dict) -> dict:
        stamped = self._cmd_queue.put(cmd)
        self._sync_queue_telemetry()
        return stamped

    def _sync_queue_telemetry(self) -> None:
        try:
            snap = self._cmd_queue.snapshot()
            self.telemetry.set_queue_depth(int(snap.get("depth_total") or 0))
            self.telemetry.set_queue_snapshot(snap)
        except Exception:
            self.telemetry.set_queue_depth(0)

    def _sync_telemetry_runtime(self, *, degraded_reason: str = "") -> None:
        from core.auxiliary_cognition import (
            active_cognition_basename,
            is_active_cognition_bundled,
        )

        self.telemetry.set_runtime_state(
            model_loaded=bool(self.model_loaded),
            degraded_reason=degraded_reason,
            active_model_basename=active_cognition_basename(),
            is_bundled_default=is_active_cognition_bundled(),
        )
        self._sync_queue_telemetry()
        try:
            self.sidecar_telemetry_updated.emit(self.telemetry.summarize())
        except Exception as e:
            logger.debug("[Sidecar] telemetry emit skipped: %s", e)

    def _unload_cognition_model(self) -> None:
        self.model = None
        self.model_loaded = False
        self._inference_transparency = {}
        gc.collect()

    def _load_cognition_model(self, path: str) -> tuple[bool, str]:
        Llama = get_llama_class()
        if Llama is None:
            return False, "llama_cpp_unavailable"
        if not path or not os.path.isfile(path):
            return False, "model_not_found"

        self._unload_cognition_model()
        n_ctx = cognition_n_ctx_for_path(path)
        self.active_chat_format = resolve_cognition_chat_format(path)
        try:
            logger.info(
                "[Sidecar] Loading cognition model on CPU (%s) n_ctx=%d format=%s",
                path,
                n_ctx,
                self.active_chat_format,
            )
            self.model = Llama(
                model_path=path,
                n_gpu_layers=0,
                n_ctx=n_ctx,
                verbose=False,
            )
            self.model_loaded = True
            self.active_model_path = path
            try:
                snap = snapshot_from_loaded_llama(
                    self.model,
                    model_path=path,
                    requested_n_gpu_layers=0,
                    n_ctx=n_ctx,
                    n_threads=max(1, int(os.cpu_count() or 4)),
                    role="sidecar",
                )
                snap["backend"] = "cpu"
                snap["compute_mode"] = "cpu"
                snap["chat_format"] = self.active_chat_format
                self._inference_transparency = snap
                log_inference_transparency(logger, role="Sidecar", snapshot=snap)
            except Exception as e:
                logger.debug("[Sidecar] inference transparency capture failed: %s", e)
                self._inference_transparency = {
                    "loaded": True,
                    "role": "sidecar",
                    "model_basename": os.path.basename(path),
                    "backend": "cpu",
                    "compute_mode": "cpu",
                }
            return True, "ok"
        except Exception as e:
            logger.error("[Sidecar] Load failed: %s", e)
            self._unload_cognition_model()
            self.active_model_path = ""
            self._inference_transparency = {
                "loaded": False,
                "role": "sidecar",
                "backend": "cpu",
                "compute_mode": "cpu",
                "degraded_reason": str(e),
            }
            return False, str(e)

    def _do_reload(self) -> None:
        self._reloading = True
        try:
            path = resolve_active_cognition_path()
            ok, msg = self._load_cognition_model(path)
            if ok:
                self._sync_telemetry_runtime()
            else:
                self._sync_telemetry_runtime(degraded_reason=msg)
            self.model_reload_finished.emit(ok, msg)
        finally:
            self._reloading = False

    def _try_load_cognition_model_if_needed(self) -> tuple[bool, str]:
        if self.model_loaded:
            return True, ""
        path = resolve_active_cognition_path()
        if not os.path.isfile(path):
            if not self._warned_missing:
                logger.warning("[Sidecar] Model not found at %s — sidecar disabled", path)
                self._warned_missing = True
            self._sync_telemetry_runtime(degraded_reason="model_not_found")
            return False, "model_not_found"
        ok, msg = self._load_cognition_model(path)
        if ok:
            self._sync_telemetry_runtime()
        else:
            self._sync_telemetry_runtime(degraded_reason=msg)
        return ok, msg

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except Exception as exc:
                if _queue_mod is not None and isinstance(exc, _queue_mod.Empty):
                    self._sync_queue_telemetry()
                    continue
                raise

            cmd["_queue_wait_ms"] = _queue_wait_ms(cmd)
            self._sync_queue_telemetry()
            op = cmd.get("op")
            if op == "shutdown":
                break
            if op == "reload":
                self._do_reload()
                continue
            if self._reloading:
                self._fail_command(cmd, "reloading")
                continue
            if not self.model_loaded:
                ok, msg = self._try_load_cognition_model_if_needed()
                if not ok:
                    self._fail_command(cmd, msg or "model_unavailable")
                    continue
            try:
                if op == "title":
                    self._do_title(cmd)
                elif op == "task":
                    self._do_task(cmd)
                elif op == "raw":
                    self._do_raw(cmd)
                elif op == "ingest_blurb":
                    self._do_ingest_blurb(cmd)
                elif op == "companion_line":
                    self._do_companion_line(cmd)
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
                                queue_wait_ms=float(cmd.get("_queue_wait_ms") or 0),
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

        self._unload_cognition_model()
        self.active_model_path = ""

    def _run_degraded_queue_loop(self) -> None:
        """Drain queue with failures so waiters never block forever."""
        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except Exception as exc:
                if _queue_mod is not None and isinstance(exc, _queue_mod.Empty):
                    continue
                raise
            if cmd.get("op") == "shutdown":
                break
            if cmd.get("op") == "reload":
                self._do_reload()
                continue
            self._fail_command(cmd, "model_unavailable")

    def _fail_command(self, cmd: dict, reason: str) -> None:
        op = cmd.get("op")
        task = cmd.get("task")
        wait_ms = float(cmd.get("_queue_wait_ms") or _queue_wait_ms(cmd))
        if op == "task" and task is not None:
            self.telemetry.record(
                task,
                ok=False,
                latency_ms=0.0,
                wait_ms=wait_ms,
                foreground=task in (SidecarTask.query_rewrite, SidecarTask.source_digest),
                reason=reason,
            )
        elif op == "title":
            self.telemetry.record(
                SidecarTask.title,
                ok=False,
                latency_ms=0.0,
                wait_ms=wait_ms,
                foreground=False,
                reason=reason,
            )
        elif op == "ingest_blurb":
            self.telemetry.record(
                SidecarTask.ingest_blurb,
                ok=False,
                latency_ms=0.0,
                wait_ms=wait_ms,
                foreground=False,
                reason=reason,
            )
        elif op == "companion_line":
            self.telemetry.record(
                SidecarTask.companion_line,
                ok=False,
                latency_ms=0.0,
                wait_ms=wait_ms,
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
                        queue_wait_ms=wait_ms,
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

    def _prepare_sidecar_prompt(self, prompt: str) -> str:
        return apply_qwen3_no_think_to_prompt(prompt, self.active_model_path)

    def _complete_prompt(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> str:
        if not self.model:
            return ""
        prompt = self._prepare_sidecar_prompt(prompt)
        stops = cognition_stop_tokens(self.active_chat_format)
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stops,
            )
            return (output.get("choices") or [{}])[0].get("text") or ""
        except Exception as e:
            logger.debug("[Sidecar] inference error: %s", e)
            return ""

    def _do_task(self, cmd: dict) -> None:
        task: SidecarTask = cmd["task"]
        payload = cmd.get("payload") or {}
        wait_ms = float(cmd.get("_queue_wait_ms") or 0.0)
        t0 = time.perf_counter()
        params = task_inference_params(task)
        prompt = build_prompt_for_task(
            task,
            chat_format=self.active_chat_format,
            model_path=self.active_model_path,
            **payload,
        )
        raw = self._complete_prompt(
            prompt,
            max_tokens=int(params.get("max_tokens", 128)),
            temperature=float(params.get("temperature", 0.2)),
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
        result = parse_task_output(task, raw, **payload)
        result.queue_wait_ms = wait_ms
        result.inference_ms = inference_ms
        out = cmd.get("out")
        if isinstance(out, list):
            out.append(result)
        ev = cmd.get("done_event")
        if ev is not None:
            ev.set()
        self._sync_telemetry_runtime()

    def _do_raw(self, cmd: dict) -> None:
        t0 = time.perf_counter()
        raw = self._complete_prompt(
            cmd.get("prompt") or "",
            max_tokens=int(cmd.get("max_tokens", 256)),
            temperature=float(cmd.get("temperature", 0.2)),
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
        wait_ms = float(cmd.get("_queue_wait_ms") or 0.0)
        self.telemetry.record(
            "raw_prompt",
            ok=bool(raw),
            latency_ms=inference_ms,
            wait_ms=wait_ms,
            foreground=False,
            reason="" if raw else "empty",
        )
        out = cmd.get("out")
        if isinstance(out, list):
            out.append(raw)
        ev = cmd.get("done_event")
        if ev is not None:
            ev.set()

    def _do_title(self, cmd: dict) -> None:
        session_id = str(cmd.get("session_id") or "")
        # Raw DB / UI text may include composer @ tokens; titling normalizes at this boundary.
        user_prompt = normalize_title_user_prompt(cmd.get("user_prompt") or "")
        assistant_reply = cmd.get("assistant_reply") or ""
        wait_ms = float(cmd.get("_queue_wait_ms") or 0.0)
        profile = get_title_profile(get_sidecar_title_inference_profile())
        context_mode = get_sidecar_title_context_mode()
        run = run_title_generation(
            self.model,
            profile=profile,
            user_prompt=user_prompt,
            assistant_reply=assistant_reply,
            context_mode=context_mode,
            model_path=self.active_model_path,
            chat_format=self.active_chat_format,
            session_id=session_id,
        )
        log_title_experiment_run(run)
        inference_ms = run.inference_ms
        raw_title = run.raw_model_output
        new_title = run.final_title
        ok = bool(new_title)
        selection = run.selection or {}
        if ok:
            logger.info(
                "[Sidecar] Title session=%s profile=%s context=%s title=%r path=%s "
                "source=%s score=%.1f runner_up=%r (%s %.1f) raw=%r",
                session_id,
                run.profile_id,
                run.context_mode,
                new_title,
                selection.get("path") or "",
                selection.get("winner_source") or "",
                float(selection.get("winner_score") or 0.0),
                selection.get("runner_up") or "",
                selection.get("runner_up_source") or "",
                float(selection.get("runner_up_score") or 0.0),
                (raw_title or "").strip().replace("\n", " ")[:120],
            )
        if new_title and self.db and session_id:
            if self.db.rename_session(session_id, new_title):
                self.title_generated.emit(session_id, new_title)
            else:
                logger.warning(
                    "[Sidecar] rename_session failed session=%s title=%r",
                    session_id,
                    new_title,
                )
        elif not ok:
            snippet = (raw_title or "").strip().replace("\n", " ")[:120]
            logger.info(
                "[Sidecar] Title empty for session=%s profile=%s raw=%r",
                session_id,
                run.profile_id,
                snippet,
            )
        self.telemetry.record(
            SidecarTask.title,
            ok=ok,
            latency_ms=inference_ms,
            wait_ms=wait_ms,
            foreground=False,
            reason="" if ok else "empty",
            meta={
                "profile": run.profile_id,
                "context_mode": run.context_mode,
                "fallback_repair": run.used_fallback_repair,
                "model_rejected": run.model_output_rejected,
                "think_stripped": run.think_block_stripped,
            },
        )
        self._sync_telemetry_runtime()

    def _do_ingest_blurb(self, cmd: dict) -> None:
        filename = str(cmd.get("filename") or "")
        sample = cmd.get("sample_text") or ""
        wait_ms = float(cmd.get("_queue_wait_ms") or 0.0)
        t0 = time.perf_counter()
        result = parse_task_output(
            SidecarTask.ingest_blurb,
            self._complete_prompt(
                build_prompt_for_task(
                    SidecarTask.ingest_blurb,
                    chat_format=self.active_chat_format,
                    model_path=self.active_model_path,
                    sample_text=sample,
                ),
                max_tokens=48,
                temperature=0.2,
            ),
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
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
            latency_ms=inference_ms,
            wait_ms=wait_ms,
            foreground=False,
            reason="" if ok else "empty",
        )
        self._sync_telemetry_runtime()

    def _do_companion_line(self, cmd: dict) -> None:
        payload = dict(cmd.get("payload") or {})
        trigger = str(payload.get("trigger") or "idle")
        wait_ms = float(cmd.get("_queue_wait_ms") or 0.0)
        t0 = time.perf_counter()
        params = task_inference_params(SidecarTask.companion_line)
        prompt = build_prompt_for_task(
            SidecarTask.companion_line,
            chat_format=self.active_chat_format,
            model_path=self.active_model_path,
            **payload,
        )
        raw = self._complete_prompt(
            prompt,
            max_tokens=int(params.get("max_tokens", 64)),
            temperature=float(params.get("temperature", 0.35)),
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
        result = parse_task_output(SidecarTask.companion_line, raw, **payload)
        ok = bool(result.ok)
        if ok:
            kind = str((result.parsed or {}).get("kind") or "idle_quip")
            line = str(result.text or "").strip()
            if line:
                self.companion_line_ready.emit(line, kind, trigger)
        self.telemetry.record(
            SidecarTask.companion_line,
            ok=ok,
            latency_ms=inference_ms,
            wait_ms=wait_ms,
            foreground=False,
            reason="" if ok else (result.error or "skip"),
        )
        self._sync_telemetry_runtime()
