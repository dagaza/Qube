"""
Qwen3 sidecar inference helpers: chat-template kwargs and completion diagnostics.

llama-cpp-python 0.3.x does not expose ``chat_template_kwargs`` on ``create_chat_completion``
directly; the server wraps ``chat_handler`` to forward kwargs into Jinja template rendering.
We use the same pattern here for ``enable_thinking=False``.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from core.qwen3_thinking_policy import is_qwen3_model, template_kwargs_for_thinking_policy

logger = logging.getLogger("Qube.Qwen3SidecarInference")


@dataclass
class CompletionDiagnostics:
    """Termination and token metadata from a single llama.cpp completion."""

    path: str = ""
    stop_sequences: list[str] = field(default_factory=list)
    finish_reason: str = ""
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    eos_encountered: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generation_error: str = ""
    raw_output: str = ""
    raw_output_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attach_chat_template_kwargs(model: Any, template_kwargs: dict[str, Any]) -> None:
    """Wrap ``model.chat_handler`` so Jinja chat templates receive extra kwargs."""
    if not template_kwargs:
        return
    try:
        import llama_cpp.llama_chat_format as llama_chat_format
    except ImportError:
        llama_chat_format = None

    base_handler = getattr(model, "chat_handler", None)
    if base_handler is None and llama_chat_format is not None:
        base_handler = (
            model._chat_handlers.get(model.chat_format)
            or llama_chat_format.get_chat_completion_handler(model.chat_format)
        )
    if base_handler is None:
        return

    def handler_with_kwargs(*args: Any, **kwargs: Any):
        merged = {**template_kwargs, **kwargs}
        return base_handler(*args, **merged)

    model.chat_handler = handler_with_kwargs


def _diagnostics_from_completion_output(
    output: dict[str, Any],
    *,
    path: str,
    stop_sequences: list[str],
    chat_template_kwargs: dict[str, Any] | None = None,
    generation_error: str = "",
) -> CompletionDiagnostics:
    choice = (output.get("choices") or [{}])[0]
    usage = output.get("usage") or {}
    finish = str(choice.get("finish_reason") or "")
    text = choice.get("text")
    if text is None:
        message = choice.get("message") or {}
        text = message.get("content") or ""
    raw = str(text or "")
    return CompletionDiagnostics(
        path=path,
        stop_sequences=list(stop_sequences),
        finish_reason=finish,
        completion_tokens=int(usage.get("completion_tokens") or 0),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        eos_encountered=finish == "stop",
        chat_template_kwargs=dict(chat_template_kwargs or {}),
        generation_error=generation_error,
        raw_output=raw,
        raw_output_length=len(raw),
    )


def raw_prompt_complete(
    model: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    stop: list[str],
    sampling_extra: dict[str, Any] | None = None,
) -> tuple[str, CompletionDiagnostics]:
    """Run ``Llama(prompt, ...)`` and capture termination metadata."""
    extra = dict(sampling_extra or {})
    extra.pop("max_tokens", None)
    extra.pop("temperature", None)
    extra.pop("chat_template_kwargs", None)
    try:
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **extra,
        )
        if hasattr(output, "__next__"):
            output = next(iter(output))
        diag = _diagnostics_from_completion_output(
            output,
            path="raw",
            stop_sequences=stop,
        )
        return diag.raw_output, diag
    except Exception as exc:
        return "", CompletionDiagnostics(
            path="raw",
            stop_sequences=list(stop),
            generation_error=str(exc),
        )


def chat_completion_complete(
    model: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    stop: list[str] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, CompletionDiagnostics]:
    """Run ``create_chat_completion`` with optional template kwargs via handler wrap."""
    if not hasattr(model, "create_chat_completion"):
        return "", CompletionDiagnostics(
            path="chat",
            generation_error="create_chat_completion_unavailable",
        )

    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if min_p is not None:
        kwargs["min_p"] = min_p
    if stop:
        kwargs["stop"] = stop

    template_kwargs = dict(chat_template_kwargs or {})
    prior_handler = getattr(model, "chat_handler", None)
    try:
        if template_kwargs:
            attach_chat_template_kwargs(model, template_kwargs)
        output = model.create_chat_completion(**kwargs)
        if hasattr(output, "__next__"):
            output = next(iter(output))
        diag = _diagnostics_from_completion_output(
            output,
            path="chat",
            stop_sequences=list(stop or []),
            chat_template_kwargs=template_kwargs,
        )
        return diag.raw_output.strip(), diag
    except Exception as exc:
        return "", CompletionDiagnostics(
            path="chat",
            stop_sequences=list(stop or []),
            chat_template_kwargs=template_kwargs,
            generation_error=str(exc),
        )
    finally:
        model.chat_handler = prior_handler


def llama_cpp_supports_template_kwargs_via_handler() -> bool:
    """True when we can inject kwargs through chat_handler wrapping."""
    try:
        import llama_cpp

        return hasattr(llama_cpp.Llama, "create_chat_completion")
    except ImportError:
        return False


__all__ = [
    "CompletionDiagnostics",
    "attach_chat_template_kwargs",
    "chat_completion_complete",
    "is_qwen3_model",
    "llama_cpp_supports_template_kwargs_via_handler",
    "raw_prompt_complete",
    "template_kwargs_for_thinking_policy",
]
