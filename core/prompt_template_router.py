"""
Single place for native prompt *representation* (reconstruction + policy overlays + stop list for logs).

Separates template structure (chat_format / Jinja) from execution policy (thinking overlays).
The native engine calls ``build_prompt_bundle`` and passes the resulting prompt string to
``Llama.create_completion(prompt=...)`` (see ``NativeLlamaEngine``).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from core.execution_policy import ExecutionPolicy
from core.native_llama_inference import native_chat_completion_kwargs
from core.model_override_store import get_override
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
    "[/INST]",
)

_MISTRAL_CLOSE_INST_RE = re.compile(r"\[/INST\]\s*$")


def _prompt_has_generation_anchor(prompt: str, template_type: str) -> bool:
    """True when the formatted prompt already opens the assistant generation slot."""
    p = (prompt or "").strip()
    if not p:
        return False
    tt = (template_type or "fallback").lower()
    if tt == "mistral":
        return bool(_MISTRAL_CLOSE_INST_RE.search(p)) or p.endswith("[INST]")
    return p.endswith(_ASSISTANT_ANCHOR_SUFFIXES)


def _maybe_append_assistant_anchor(
    prompt: str,
    template_type: str,
) -> tuple[str, bool]:
    """
    Append a template-safe generation anchor only when missing.

    Mistral instruct prompts must end at [/INST]; never append Phi-style <|assistant|>.
    """
    if _prompt_has_generation_anchor(prompt, template_type):
        return prompt, False
    tt = (template_type or "fallback").lower()
    if tt == "mistral":
        return prompt, False
    return (prompt or "") + "\n<|assistant|>\n", True


def _insert_before_last_anchor(prompt: str, anchor: str, text: str) -> str:
    p = prompt or ""
    idx = p.rfind(anchor)
    if idx < 0:
        return p + "\n" + text
    return p[:idx] + text.rstrip() + "\n" + p[idx:]


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
        p, _ = _maybe_append_assistant_anchor(p or "", override.template_type)
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

    if reasoning_mode != "soft" and reasoning_mode != "disabled":
        return prompt

    tt = (template_type or "fallback").lower()
    suffix = (
        "Write only the user-facing response."
        if reasoning_mode == "disabled"
        else None
    )

    if tt == "chatml":
        if reasoning_mode == "soft":
            return _insert_before_last_anchor(
                prompt or "",
                "<|im_start|>assistant",
                "You may use <think>...</think> internally. "
                "Write only the user-facing response outside those tags.",
            )
        return _insert_before_last_anchor(
            prompt or "",
            "<|im_start|>assistant",
            suffix or "Write only the user-facing response.",
        )

    if tt == "llama3":
        anchor = "<|start_header_id|>assistant<|end_header_id|>"
        if reasoning_mode == "soft":
            return _insert_before_last_anchor(
                prompt or "",
                anchor,
                "Keep any hidden reasoning private and write only the user-facing response.",
            )
        return _insert_before_last_anchor(
            prompt or "",
            anchor,
            suffix or "Write only the user-facing response.",
        )

    if tt == "phi":
        p = prompt or ""
        if reasoning_mode == "soft" and _PHI_ASSISTANT in p:
            prefix = "Keep hidden reasoning private. Write only the user-facing response."
            before, sep, after = p.rpartition(_PHI_ASSISTANT)
            if sep:
                return before + prefix + _PHI_ASSISTANT + after
        return (prompt or "") + "\n" + (suffix or "Write only the user-facing response.")

    if tt == "mistral":
        if reasoning_mode == "soft":
            return (prompt or "") + "\n(Use internal reasoning. Do not expose it.)"
        return (prompt or "") + "\n" + (suffix or "Write only the user-facing response.")

    return (prompt or "") + "\n" + (suffix or "Write only the user-facing response.")


def build_prompt_bundle(
    llama: Any,
    messages: list[dict],
    model_profile: Optional["ModelReasoningProfile"],
    execution_policy: ExecutionPolicy,
    *,
    effective_chat_format: Optional[str] = None,
    suppress_gguf_metadata: bool = False,
    prompt_contract_stops: Optional[Sequence[str]] = None,
    publisher_guidance: Optional[Any] = None,
) -> tuple[RenderPromptBundle, str, Any]:
    """
    Build RenderPromptBundle using existing reconstruct_formatted_prompt + policy overlays + stops.

    ``model_profile`` is used for learned override lookup; pass through from detection.
    When called from the native engine, pass ``effective_chat_format`` and
    ``suppress_gguf_metadata`` in lockstep with ``PromptContract`` + unsafe-template policy.
    ``prompt_contract_stops`` (static family stops) are merged at the end so the bundle
    matches ``PromptContract.stop``.
    """
    _cc_kw = native_chat_completion_kwargs(llama)
    prompt_txt, fmt_stop, recon_note = reconstruct_formatted_prompt(
        llama,
        messages,
        effective_chat_format=effective_chat_format,
        suppress_gguf_metadata=suppress_gguf_metadata,
    )
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
    # Thinking-tag stops are for ChatML/Llama3/Phi where tags are in-vocab anchors.
    # On Jinja/GGUF templates they can truncate the first token to empty (stream + non-stream).
    if reasoning_mode == "disabled" and template_type in ("chatml", "llama3", "phi"):
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
    learned = get_override(
        model_profile.model_name if model_profile else "unknown"
    )
    if learned:
        if learned.extra_stop_tokens:
            merged_learned, _ = merge_stop_lists(
                bundle.stop_tokens, learned.extra_stop_tokens
            )
            bundle.stop_tokens = list(merged_learned)
        if learned.enforce_assistant_anchor:
            bundle.prompt, _ = _maybe_append_assistant_anchor(
                bundle.prompt or "", template_type
            )
        logger.info(
            "[LLM-SELF-HEAL-APPLY] model=%s stops=%d anchor=%s",
            learned.model_name,
            len(learned.extra_stop_tokens),
            learned.enforce_assistant_anchor,
        )
    if prompt_contract_stops:
        merged_cc, _ = merge_stop_lists(
            bundle.stop_tokens, list(prompt_contract_stops)
        )
        bundle.stop_tokens = list(merged_cc)
    if publisher_guidance is not None:
        pg_tags = getattr(publisher_guidance, "thinking_tags", None) or ()
        if pg_tags:
            merged_pg, _ = merge_stop_lists(bundle.stop_tokens, list(pg_tags))
            bundle.stop_tokens = list(merged_pg)
            logger.info(
                "[LLM-README-GUIDANCE] source=%s tags=%d default_without_system=%s",
                getattr(publisher_guidance, "source", ""),
                len(pg_tags),
                getattr(publisher_guidance, "default_reasoning_without_system", "unknown"),
            )
    logger.info(
        "[LLM-PROMPT-ROUTER] template=%s reasoning=%s stop_count=%d",
        template_type,
        reasoning_mode,
        len(bundle.stop_tokens),
    )
    return bundle, recon_note or "", fmt_stop
