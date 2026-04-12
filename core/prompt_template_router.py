"""
Single place for native prompt *representation* (reconstruction + policy overlays + stop list for logs).

Separates template structure (chat_format / Jinja) from execution policy (thinking overlays).
Does not alter llama-cpp tokenizer or create_chat_completion message path — bundle is used for
validation, logging, and trace prefetch unless inference is explicitly wired to it later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from core.execution_policy import ExecutionPolicy
from core.native_llama_inference import native_chat_completion_kwargs
from core.native_llm_debug import merge_stop_lists, reconstruct_formatted_prompt

if TYPE_CHECKING:
    from core.model_reasoning_profile import ModelReasoningProfile

logger = logging.getLogger("Qube.PromptTemplateRouter")

# Policy-only extra stops when we instruct the model not to emit reasoning (disabled mode).
_POLICY_DISABLED_EXTRA_STOPS: tuple[str, ...] = (
    "<think>",
    "</think>",
    "<thinking>",
    "</thinking>",
)

_SYSTEM_OVERLAY_DISABLED = (
    "Do not output reasoning or internal thoughts. Output final answer only."
)

_SYSTEM_OVERLAY_SOFT = (
    "You may use <redacted_thinking>...</redacted_thinking> for reasoning. "
    "Only final answer outside."
)


@dataclass
class RenderPromptBundle:
    prompt: str
    chat_format: str
    stop_tokens: List[str]
    template_type: str  # chatml, llama3, mistral, jinja, fallback
    reasoning_mode: str  # disabled | soft | hard


def infer_template_type(llama: Any) -> str:
    """Classify template routing key from GGUF metadata + chat_format."""
    md = getattr(llama, "metadata", None) or {}
    tmpl = md.get("tokenizer.chat_template")
    if isinstance(tmpl, str) and tmpl.strip():
        return "jinja"
    cf = (getattr(llama, "chat_format", None) or "").strip()
    if cf == "chatml":
        return "chatml"
    if cf == "llama-3":
        return "llama3"
    if cf == "mistral-instruct":
        return "mistral"
    return "fallback"


def resolve_reasoning_mode(policy: ExecutionPolicy) -> str:
    """
    Map ExecutionPolicy to overlay mode for prompt suffixes.
    hard → no overlay; soft + allow thinking → soft overlay; else → disabled overlay.
    """
    if policy.enforcement_mode == "hard":
        return "hard"
    if policy.allow_thinking_tokens:
        return "soft"
    return "disabled"


def _apply_reasoning_overlay(base_prompt: Optional[str], reasoning_mode: str) -> str:
    if not base_prompt:
        return ""
    if reasoning_mode == "hard":
        return base_prompt
    if reasoning_mode == "disabled":
        return base_prompt + "\n\n" + _SYSTEM_OVERLAY_DISABLED
    if reasoning_mode == "soft":
        return base_prompt + "\n\n" + _SYSTEM_OVERLAY_SOFT
    return base_prompt


def build_prompt_bundle(
    llama: Any,
    messages: list[dict],
    model_profile: Optional["ModelReasoningProfile"],
    execution_policy: ExecutionPolicy,
) -> tuple[RenderPromptBundle, str, Any]:
    """
    Build RenderPromptBundle using existing reconstruct_formatted_prompt + policy overlays + stops.

    ``model_profile`` is reserved for future template routing; pass through from detection.
    """
    _ = model_profile
    _cc_kw = native_chat_completion_kwargs(llama)
    prompt_txt, fmt_stop, recon_note = reconstruct_formatted_prompt(llama, messages)
    template_type = infer_template_type(llama)
    reasoning_mode = resolve_reasoning_mode(execution_policy)
    prompt_final = _apply_reasoning_overlay(prompt_txt, reasoning_mode)

    merged, _ = merge_stop_lists(_cc_kw.get("stop"), fmt_stop)
    stops = list(merged)
    if reasoning_mode == "disabled":
        stops = stops + list(_POLICY_DISABLED_EXTRA_STOPS)

    cf = str(getattr(llama, "chat_format", "") or "")
    bundle = RenderPromptBundle(
        prompt=prompt_final,
        chat_format=cf,
        stop_tokens=stops,
        template_type=template_type,
        reasoning_mode=reasoning_mode,
    )
    logger.info(
        "[LLM-PROMPT-ROUTER] template=%s reasoning=%s stop_count=%d",
        template_type,
        reasoning_mode,
        len(stops),
    )
    return bundle, recon_note or "", fmt_stop
