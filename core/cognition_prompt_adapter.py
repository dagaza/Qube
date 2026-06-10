"""
Chat templates and stop tokens for auxiliary cognition models.
"""
from __future__ import annotations

import logging
import os

from core.app_settings import get_sidecar_chat_format

logger = logging.getLogger("Qube.CognitionPromptAdapter")

IM_END = "<|im_end|>"
CHATML_SYSTEM_MARKER = "<|im_start|>system\n"
QWEN3_NO_THINK_DIRECTIVE = "/no_think"
SUPPORTED_FORMATS = ("chatml", "llama-3", "phi", "gemma")


def is_qwen3_cognition_model(model_path: str) -> bool:
    """True when the active sidecar GGUF is a Qwen3-family checkpoint."""
    name = os.path.basename(model_path or "").lower()
    return "qwen3" in name or "qwen-3" in name


def finalize_cognition_system_prompt(system: str, model_path: str = "") -> str:
    """Append Qwen3 ``/no_think`` to system text for non-thinking sidecar tasks."""
    text = (system or "").strip()
    if not text or not is_qwen3_cognition_model(model_path):
        return system
    low = text.lower()
    if QWEN3_NO_THINK_DIRECTIVE in low or "/think" in low:
        return system
    return f"{text} {QWEN3_NO_THINK_DIRECTIVE}"


def _patch_chatml_system_no_think(prompt: str) -> str:
    marker = CHATML_SYSTEM_MARKER
    if marker not in prompt:
        return prompt
    idx = prompt.find(marker)
    rest = prompt[idx + len(marker) :]
    end = rest.find(IM_END)
    if end < 0:
        return prompt
    body = rest[:end].rstrip()
    if QWEN3_NO_THINK_DIRECTIVE in body.lower() or "/think" in body.lower():
        return prompt
    new_body = f"{body} {QWEN3_NO_THINK_DIRECTIVE}"
    return prompt[: idx + len(marker)] + new_body + rest[end:]


def apply_qwen3_no_think_to_prompt(prompt: str, model_path: str = "") -> str:
    """Apply ``/no_think`` to rendered ChatML or flat sidecar prompts."""
    text = prompt or ""
    if not text.strip() or not is_qwen3_cognition_model(model_path):
        return text
    if QWEN3_NO_THINK_DIRECTIVE in text.lower() or "/think" in text.lower():
        return text
    if CHATML_SYSTEM_MARKER in text:
        return _patch_chatml_system_no_think(text)
    return f"{QWEN3_NO_THINK_DIRECTIVE}\n\n{text.lstrip()}"


def infer_format_from_path(model_path: str) -> str:
    name = os.path.basename(model_path or "").lower()
    if "phi" in name:
        return "phi"
    if "gemma" in name:
        return "gemma"
    if any(tok in name for tok in ("llama-3", "llama3", "llama_3")):
        return "llama-3"
    if any(
        tok in name
        for tok in ("qwen3", "qwen-3", "qwen2", "qwen2.5", "qwen-2", "chatml", "mistral")
    ):
        return "chatml"
    logger.debug("[Cognition] unknown model name %s — defaulting to chatml", name)
    return "chatml"


def resolve_cognition_chat_format(
    model_path: str,
    setting_override: str | None = None,
) -> str:
    override = (setting_override or get_sidecar_chat_format() or "auto").lower().strip()
    if override != "auto" and override in SUPPORTED_FORMATS:
        return override
    return infer_format_from_path(model_path)


def build_cognition_prompt(
    system: str,
    user: str,
    fmt: str,
    *,
    model_path: str = "",
) -> str:
    system = finalize_cognition_system_prompt(system, model_path)
    f = (fmt or "chatml").lower()
    if f == "llama-3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if f == "phi":
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            "<|assistant|>\n"
        )
    if f == "gemma":
        return (
            f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    return (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user}{IM_END}\n"
        "<|im_start|>assistant\n"
    )


def cognition_stop_tokens(fmt: str) -> list[str]:
    f = (fmt or "chatml").lower()
    if f == "llama-3":
        return ["<|eot_id|>", "<|end_of_text|>", "\n\n"]
    if f == "phi":
        return ["<|end|>", "\n\n"]
    if f == "gemma":
        return ["<end_of_turn>", "\n\n"]
    return [IM_END, "\n\n"]
