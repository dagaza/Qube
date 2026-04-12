"""
Single place for native prompt *representation* (reconstruction + policy overlays + stop list for logs).

Separates template structure (chat_format / Jinja) from execution policy (thinking overlays).
Does not alter llama-cpp tokenizer or create_chat_completion message path — bundle is used for
validation, logging, and trace prefetch unless inference is explicitly wired to it later.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.execution_policy import ExecutionPolicy
from core.native_llama_inference import native_chat_completion_kwargs
from core.native_llm_debug import merge_stop_lists, reconstruct_formatted_prompt
from core.template_override import TemplateOverride, detect_template_override

if TYPE_CHECKING:
    from core.model_reasoning_profile import ModelReasoningProfile

logger = logging.getLogger("Qube.PromptTemplateRouter")

# Policy-only extra stops when we instruct the model not to emit reasoning (disabled mode).
_POLICY_DISABLED_EXTRA_STOPS: tuple[str, ...] = (
    "<redacted_thinking>",
    "</redacted_thinking>",
    "<thinking>",
    "</thinking>",
)

_PHI_ASSISTANT = "<|assistant|>"

_ASSISTANT_ANCHOR_SUFFIXES: tuple[str, ...] = (
    "<|assistant|>",
    "<|im_start|>assistant",
    "[INST]",
)


def _llama_display_name(llama: Any) -> str:
    md = getattr(llama, "metadata", None) or {}
    if isinstance(md, dict):
        for k in ("general.name", "general.basename", "name"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    mp = getattr(llama, "model_path", None)
    if isinstance(mp, str) and mp:
        return os.path.basename(mp)
    return ""


def _tokenizer_info_dict(llama: Any) -> Dict[str, Any]:
    md = getattr(llama, "metadata", None) or {}
    if not isinstance(md, dict):
        return {}
    return {"tokenizer.chat_template": md.get("tokenizer.chat_template")}


def _apply_template_override(bundle: "RenderPromptBundle", override: TemplateOverride) -> None:
    merged, _ = merge_stop_lists(bundle.stop_tokens, override.extra_stops)
    bundle.stop_tokens = list(merged)
    p = bundle.prompt
    if override.enforce_assistant_anchor:
        s = (p or "").strip()
        if not s.endswith(_ASSISTANT_ANCHOR_SUFFIXES):
            p = (p or "") + "\n<|assistant|>\n"
            bundle.prompt = p
    if override.force_prefix:
        bundle.prompt = bundle.prompt + override.force_prefix


@dataclass
class RenderPromptBundle:
    prompt: str
    chat_format: str
    stop_tokens: List[str]
    template_type: str  # chatml, llama3, phi, mistral, jinja, fallback
    reasoning_mode: str  # disabled | soft | hard


def infer_template_type(llama: Any) -> str:
    """Classify template routing key from GGUF tokenizer.chat_template string and chat_format."""
    md = getattr(llama, "metadata", None) or {}
    tmpl = md.get("tokenizer.chat_template")
    if isinstance(tmpl, str) and tmpl.strip():
        t = tmpl
        if "<|im_start|>" in t:
            return "chatml"
        if "start_header_id" in t:
            return "llama3"
        if "<|system|>" in t and "<|assistant|>" in t:
            return "phi"
        if "[INST]" in t:
            return "mistral"
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


def apply_reasoning_injection(prompt: str, template_type: str, reasoning_mode: str) -> str:
    """
    Inject reasoning instructions SAFELY based on template type.
    Must NEVER break assistant anchor or template structure.
    """
    if reasoning_mode == "hard":
        return prompt

    if reasoning_mode == "disabled":
        return (prompt or "") + "\nRespond with final answer only."

    if reasoning_mode != "soft":
        return prompt

    tt = (template_type or "fallback").lower()

    if tt == "chatml":
        return (prompt or "") + (
            "\nYou may use <redacted_thinking>...</redacted_thinking> internally. "
            "Only output final answer."
        )

    if tt == "llama3":
        return (prompt or "") + "\nUse hidden reasoning if needed. Only output final answer."

    if tt == "phi":
        p = prompt or ""
        if _PHI_ASSISTANT in p:
            prefix = "Use hidden reasoning internally. Only output final answer."
            before, sep, after = p.rpartition(_PHI_ASSISTANT)
            if sep:
                return before + prefix + _PHI_ASSISTANT + after
        return (prompt or "") + "\nOnly output final answer."

    if tt == "mistral":
        return (prompt or "") + "\n(Use internal reasoning. Do not expose it.)"

    return (prompt or "") + "\nOnly output final answer."


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

    logger.info(
        "[LLM-TEMPLATE] type=%s reasoning=%s",
        template_type,
        reasoning_mode,
    )

    prompt_final = apply_reasoning_injection(
        prompt_txt or "",
        template_type,
        reasoning_mode,
    )

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
    model_name = _llama_display_name(llama)
    override = detect_template_override(model_name, _tokenizer_info_dict(llama))
    if override is not None:
        _apply_template_override(bundle, override)
        logger.info(
            "[LLM-TEMPLATE-OVERRIDE] model=%s template=%s stops_added=%d",
            model_name,
            override.template_type,
            len(override.extra_stops),
        )
    logger.info(
        "[LLM-PROMPT-ROUTER] template=%s reasoning=%s stop_count=%d",
        template_type,
        reasoning_mode,
        len(bundle.stop_tokens),
    )
    return bundle, recon_note or "", fmt_stop
