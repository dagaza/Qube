"""
Persistent QThread that owns a llama-cpp-python Llama instance.
All load, unload, and streaming inference run on this thread only.
"""
from __future__ import annotations

import gc
import logging
import os
import queue
import threading
from typing import Any, Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("Qube.NativeLLM")

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore


class NativeLlamaEngine(QThread):
    """
    Command loop: LOAD, UNLOAD, GENERATE (streaming deltas returned via token_queue).
    Signals are emitted from this thread; Qt will queue them to the UI thread.
    """

    status_update = pyqtSignal(str)
    load_finished = pyqtSignal(bool, str)  # ok, message

    def __init__(self):
        super().__init__()
        self._cmd_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._llama: Any = None
        self._model_path: Optional[str] = None
        self._n_gpu_layers: int = 0
        self._n_ctx: int = 4096
        self._n_threads: int = 1
        self._cancel_generation = False

    def stop_engine(self) -> None:
        """Request shutdown and wait for the thread to finish."""
        self._stop.set()
        self._cmd_queue.put({"op": "shutdown"})
        self.wait(30_000)

    def request_cancel_generation(self) -> None:
        self._cancel_generation = True

    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        n_threads: int,
    ) -> None:
        self._cmd_queue.put(
            {
                "op": "load",
                "path": model_path,
                "n_gpu_layers": int(n_gpu_layers),
                "n_ctx": int(n_ctx),
                "n_threads": max(1, int(n_threads)),
            }
        )

    def unload_model(self) -> None:
        self._cmd_queue.put({"op": "unload"})

    def enqueue_generation(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        token_queue: queue.Queue,
        done_event: threading.Event,
    ) -> None:
        self._cmd_queue.put(
            {
                "op": "generate",
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "token_queue": token_queue,
                "done_event": done_event,
            }
        )

    def enqueue_simple_completion(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        out: list,
        done_event: threading.Event,
    ) -> None:
        """Non-streaming completion for LLMWorker.generate() helpers (same thread as other ops)."""
        self._cmd_queue.put(
            {
                "op": "chat_once",
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "out": out,
                "done_event": done_event,
            }
        )

    def run(self) -> None:
        if Llama is None:
            logger.error("llama_cpp not available; native engine cannot start.")
            return

        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            op = cmd.get("op")
            if op == "shutdown":
                self._do_unload()
                break
            if op == "load":
                self._do_load(cmd)
            elif op == "unload":
                self._do_unload()
            elif op == "generate":
                self._do_generate(cmd)
            elif op == "chat_once":
                self._do_chat_once(cmd)

    def _do_load(self, cmd: dict) -> None:
        path = cmd.get("path") or ""
        n_gpu = int(cmd.get("n_gpu_layers", 0))
        n_ctx = int(cmd.get("n_ctx", 4096))
        n_threads = int(cmd.get("n_threads") or 0)
        if n_threads < 1:
            n_threads = max(1, int(os.cpu_count() or 4))

        if not path or not os.path.isfile(path):
            self.load_finished.emit(False, f"Model file not found: {path}")
            self.status_update.emit("Native engine: no model file")
            return

        self._do_unload()
        try:
            self.status_update.emit("Loading native model…")
            self._llama = Llama(
                model_path=path,
                n_gpu_layers=n_gpu,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
            self._model_path = path
            self._n_gpu_layers = n_gpu
            self._n_ctx = n_ctx
            self._n_threads = n_threads
            logger.info(
                "[Native] Loaded %s (n_gpu_layers=%s, n_ctx=%s, n_threads=%s)",
                path,
                n_gpu,
                n_ctx,
                n_threads,
            )
            self.load_finished.emit(True, os.path.basename(path))
            self.status_update.emit(f"Native model ready: {os.path.basename(path)}")
        except Exception as e:
            logger.exception("[Native] Load failed: %s", e)
            self._llama = None
            self._model_path = None
            self.load_finished.emit(False, str(e))
            self.status_update.emit("Native engine load failed")

    def _do_unload(self) -> None:
        if self._llama is None:
            return
        try:
            self.status_update.emit("Unloading native model…")
            # llama-cpp-python exposes .close() on Llama in recent versions
            close = getattr(self._llama, "close", None)
            if callable(close):
                close()
        except Exception as e:
            logger.debug("[Native] close(): %s", e)
        finally:
            self._llama = None
            self._model_path = None
            gc.collect()
            self.status_update.emit("Native model unloaded")
            logger.info("[Native] Model unloaded")

    def _do_generate(self, cmd: dict) -> None:
        token_queue: queue.Queue = cmd["token_queue"]
        done_event: threading.Event = cmd["done_event"]
        messages = cmd.get("messages") or []
        temperature = float(cmd.get("temperature", 0.7))
        max_tokens = int(cmd.get("max_tokens", 512))

        if self._llama is None:
            token_queue.put(("error", "Native model not loaded"))
            token_queue.put(("end", ""))
            done_event.set()
            return

        self._cancel_generation = False
        final_text = ""

        try:
            stream = self._llama.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if self._cancel_generation:
                    break
                try:
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                except Exception:
                    delta = ""
                if delta:
                    final_text += delta
                    token_queue.put(("delta", delta))

            token_queue.put(("end", final_text))
        except Exception as e:
            logger.exception("[Native] Generation error: %s", e)
            token_queue.put(("error", str(e)))
            token_queue.put(("end", final_text))
        finally:
            done_event.set()

    def _do_chat_once(self, cmd: dict) -> None:
        out: list = cmd["out"]
        done_event: threading.Event = cmd["done_event"]
        messages = cmd.get("messages") or []
        temperature = float(cmd.get("temperature", 0.1))
        max_tokens = int(cmd.get("max_tokens", 1000))

        if self._llama is None:
            out.append("")
            done_event.set()
            return

        try:
            r = self._llama.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            text = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            out.append(text)
        except Exception as e:
            logger.exception("[Native] chat_once error: %s", e)
            out.append("")
        finally:
            done_event.set()
