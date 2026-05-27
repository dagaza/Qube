"""
Chat templates and stop tokens for auxiliary cognition models.
"""
from __future__ import annotations

import logging
import os

from core.app_settings import get_sidecar_chat_format

logger = logging.getLogger("Qube.CognitionPromptAdapter")

IM_END = "<|im_end|>"
SUPPORTED_FORMATS = ("chatml", "llama-3", "phi", "gemma")


def infer_format_from_path(model_path: str) -> str:
    name = os.path.basename(model_path or "").lower()
    if "phi" in name:
        return "phi"
    if "gemma" in name:
        return "gemma"
    if any(tok in name for tok in ("llama-3", "llama3", "llama_3")):
        return "llama-3"
    if any(tok in name for tok in ("qwen2", "qwen2.5", "qwen-2", "chatml", "mistral")):
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


def build_cognition_prompt(system: str, user: str, fmt: str) -> str:
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
