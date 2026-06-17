"""Jinja ``enable_thinking`` policy helpers for thinking-capable GGUF families (no llama_cpp)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.execution_policy import ExecutionPolicy


def is_qwen3_model(*, model_path: str = "", model_name: str = "") -> bool:
    """True when the loaded GGUF identity indicates a Qwen3 family model."""
    from core.cognition_prompt_adapter import is_qwen3_cognition_model

    if is_qwen3_cognition_model(model_path):
        return True
    low = (model_name or "").lower()
    return "qwen3" in low or "qwen-3" in low


def is_nemotron_family_model(*, model_path: str = "", model_name: str = "") -> bool:
    """True when the loaded GGUF identity indicates NVIDIA Nemotron family."""
    ident = f"{model_path} {model_name}".lower()
    return "nemotron" in ident or "nvidia" in ident


def template_kwargs_for_thinking_policy(
    policy: "ExecutionPolicy",
    *,
    model_path: str = "",
    model_name: str = "",
) -> dict[str, Any]:
    """Return Jinja ``enable_thinking`` kwargs when the model template supports them."""
    if not (
        is_qwen3_model(model_path=model_path, model_name=model_name)
        or is_nemotron_family_model(model_path=model_path, model_name=model_name)
    ):
        return {}
    return {"enable_thinking": bool(policy.allow_thinking_tokens)}


__all__ = [
    "is_nemotron_family_model",
    "is_qwen3_model",
    "template_kwargs_for_thinking_policy",
]
