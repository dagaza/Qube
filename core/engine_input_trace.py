"""
Ground-truth capture of arguments passed into llama-cpp-python's completion API.

llama-cpp-python's chat path formats messages in a chat handler, then calls
``Llama.create_completion(prompt=...)`` where ``prompt`` is typically a list of
token ids (see ``chat_formatter_to_chat_completion_handler``). We wrap
``create_completion`` for the duration of ``create_chat_completion`` to record
that value without re-running Jinja or PromptContract logic.

Caveat: ``Llama._create_completion`` may still prepend/append BOS/EOS token groups
to build the final evaluated sequence. This module captures the handler→completion
boundary only.

Enable capture with ``QUBE_ENGINE_INPUT_TRACE=1`` (default off).
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, List, Optional, Union

logger = logging.getLogger("Qube.EngineInputTrace")


def engine_input_trace_enabled() -> bool:
    return os.environ.get("QUBE_ENGINE_INPUT_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class EngineInputTrace:
    """Observability-only snapshot at the llama.cpp-python engine boundary."""

    model_name: str
    timestamp: float
    input_mode: str  # "messages" or "completion"
    messages: Optional[list]
    prompt: Optional[str]
    serialized_input: Optional[str]
    chat_format: Optional[str]
    stop_tokens: list[str]
    source: str  # "llama_cpp_chat", "llama_cpp_completion"
    capture_notes: Optional[str] = None
    trace_id: int = 0


class EngineInputTracer:
    """Process-wide last-trace store (native engine runs on a single worker thread)."""

    _instance: Optional["EngineInputTracer"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "EngineInputTracer":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._data_lock = threading.Lock()
                    inst._last: Optional[EngineInputTrace] = None
                    inst._seq = 0
                    cls._instance = inst
        return cls._instance

    def log(self, trace: EngineInputTrace) -> None:
        with self._data_lock:
            self._seq += 1
            self._last = replace(trace, trace_id=self._seq)

    def get_last(self) -> Optional[EngineInputTrace]:
        with self._data_lock:
            return self._last


def detokenize_prompt_arg(
    llama: Any, prompt: Union[str, List[int], Any]
) -> tuple[Optional[str], Optional[str]]:
    """
    Turn the ``prompt`` argument passed to ``create_completion`` into a string.

    Returns (text, error_message). error_message is set on failure.
    """
    if isinstance(prompt, str):
        return prompt, None
    if not isinstance(prompt, list) or not prompt:
        return None, "prompt_not_non_empty_str_or_token_list"

    out: list[str] = []
    prev: list[int] = []
    try:
        for tid in prompt:
            if not isinstance(tid, int):
                return None, f"non_int_token_id: {type(tid).__name__}"
            b = llama.detokenize(
                [tid],
                prev_tokens=prev if prev else None,
                special=True,
            )
            piece = b.decode("utf-8", errors="replace")
            out.append(piece)
            prev.append(tid)
        return "".join(out), None
    except Exception as e:
        logger.debug("[EngineInputTrace] detokenize failed: %s", e)
        return None, str(e)


@contextmanager
def install_create_completion_capture(llama: Any) -> Iterator[dict[str, Any]]:
    """
    Temporarily wrap ``llama.create_completion`` to record the first ``prompt``
    argument (positional or keyword).
    """
    cap: dict[str, Any] = {"prompt": None, "captured": False}
    orig = llama.create_completion

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if not cap["captured"]:
            p = kwargs.get("prompt")
            if p is None and args:
                p = args[0]
            cap["prompt"] = p
            cap["captured"] = True
        return orig(*args, **kwargs)

    llama.create_completion = _wrapped  # type: ignore[method-assign]
    try:
        yield cap
    finally:
        llama.create_completion = orig  # type: ignore[method-assign]


def engine_input_trace_to_public_dict(trace: EngineInputTrace) -> dict[str, Any]:
    """JSON-safe dict for routing debug / UI (deep structures should already be plain dicts)."""
    return {
        "trace_id": trace.trace_id,
        "model_name": trace.model_name,
        "timestamp": trace.timestamp,
        "input_mode": trace.input_mode,
        "messages": trace.messages,
        "prompt": trace.prompt,
        "serialized_input": trace.serialized_input,
        "chat_format": trace.chat_format,
        "stop_tokens": list(trace.stop_tokens),
        "source": trace.source,
        "capture_notes": trace.capture_notes,
    }
