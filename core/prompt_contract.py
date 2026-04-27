from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from typing import Any, Literal, Optional

from core.native_llm_debug import reconstruct_formatted_prompt
from core.template_safety import is_unsafe_chat_template

PromptMode = Literal["messages", "rendered"]
TemplateSource = Literal["gguf", "override", "fallback", "fallback_unsafe_gguf"]

logger = logging.getLogger("Qube.PromptContract")
PromptConfidence = Literal["high", "medium", "low"]
_HARMONY_FINAL_STOPS: list[str] = [
    "<|return|>",
    "\nWe need to",
    " We need to",
    "\nWe should",
    " We should",
    "\nWe have",
    " We have",
    "\nLet's",
    " Let's",
]

_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[INST\]|\[/INST\]", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),
    re.compile(r"<\|start_header_id\|>|<\|end_header_id\|>", re.I),
)


@dataclass
class PromptContract:
    mode: PromptMode
    chat_format: Optional[str]
    prompt: Optional[str]
    messages: Optional[list[dict[str, Any]]]
    stop: list[str]
    template_source: TemplateSource
    confidence: PromptConfidence


@dataclass
class PromptContractResolution:
    contract: PromptContract
    warning: Optional[str] = None
    handler_available: bool = True
    template_safety: Optional[dict[str, Any]] = None


def assert_prompt_contract(contract: PromptContract) -> None:
    has_messages = bool(contract.messages)
    has_prompt = bool((contract.prompt or "").strip())
    if has_messages and has_prompt:
        raise ValueError("PromptContract invalid: both messages and prompt are set.")
    if not has_messages and not has_prompt:
        raise ValueError("PromptContract invalid: neither messages nor prompt are set.")
    if contract.mode == "messages" and not has_messages:
        raise ValueError("PromptContract invalid: mode=messages requires messages.")
    if contract.mode == "rendered" and not has_prompt:
        raise ValueError("PromptContract invalid: mode=rendered requires prompt.")


def _messages_payload(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        out.append({"role": m.get("role", "user"), "content": m.get("content") or ""})
    return out


def _is_gpt_oss_model(model_name: str) -> bool:
    n = (model_name or "").lower()
    return "gpt-oss" in n or ("gpt" in n and "oss" in n)


def render_harmony_final_prompt(messages: list[dict[str, Any]]) -> str:
    """
    Render OpenAI gpt-oss / Harmony prompt with assistant pre-filled in final channel.

    gpt-oss models are Harmony-trained and tend to emit free-text analysis when forced
    through ChatML. Pre-filling the assistant final channel asks for the user-facing answer
    directly while still using ``create_completion(prompt=...)``.
    """
    parts: list[str] = []
    for m in _messages_payload(messages):
        role = str(m.get("role") or "user").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"
        parts.append(f"<|start|>{role}<|message|>{content}<|end|>")
    parts.append("<|start|>assistant<|channel|>final<|message|>")
    return "\n".join(parts)


def contains_template_markers(messages: list[dict[str, Any]]) -> bool:
    for m in messages or []:
        txt = str((m or {}).get("content") or "")
        for pat in _MARKER_PATTERNS:
            if pat.search(txt):
                return True
    return False


def _model_display_name(llama: Any) -> str:
    md = getattr(llama, "metadata", None) or {}
    if isinstance(md, dict):
        for k in ("general.name", "general.basename", "name"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    mp = getattr(llama, "model_path", None)
    if isinstance(mp, str) and mp:
        return os.path.basename(mp)
    return "unknown"


def _handlers(llama: Any) -> set[str]:
    h = getattr(llama, "_chat_handlers", None) or {}
    if isinstance(h, dict):
        return set(str(k) for k in h.keys())
    return set()


def _has_chat_template(md: dict[str, Any]) -> bool:
    tmpl = md.get("tokenizer.chat_template")
    return isinstance(tmpl, str) and bool(tmpl.strip())


def stops_for_format(chat_format: Optional[str]) -> list[str]:
    cf = (chat_format or "").strip().lower()
    if cf == "llama-2":
        return ["[INST]", "[/INST]"]
    if cf == "chatml":
        return ["<|im_end|>"]
    if cf == "mistral-instruct":
        return []
    return []


def _format_supported(chat_format: str, handlers: set[str]) -> bool:
    if not chat_format:
        return False
    if chat_format in handlers:
        return True
    # Common compat mode: if handlers are unavailable in test doubles, allow known fallbacks.
    return chat_format in {"chatml", "llama-2", "llama-3", "mistral-instruct"}


def resolve_prompt_contract(llama: Any, messages: list[dict[str, Any]]) -> PromptContractResolution:
    md = getattr(llama, "metadata", None) or {}
    if not isinstance(md, dict):
        md = {}
    handlers = _handlers(llama)
    model_name = _model_display_name(llama).lower()
    msg_payload = _messages_payload(messages)

    if _is_gpt_oss_model(model_name):
        c = PromptContract(
            mode="rendered",
            chat_format=None,
            prompt=render_harmony_final_prompt(msg_payload),
            messages=None,
            stop=list(_HARMONY_FINAL_STOPS),
            template_source="fallback",
            confidence="medium",
        )
        assert_prompt_contract(c)
        return PromptContractResolution(
            contract=c,
            warning="Using Harmony final-channel rendered prompt for gpt-oss.",
            handler_available=True,
            template_safety={"unsafe": False, "reasons": []},
        )

    # Step 1: GGUF template + handler available.
    if _has_chat_template(md) and ("chat_template.default" in handlers):
        tmpl_raw = md.get("tokenizer.chat_template")
        tmpl = tmpl_raw if isinstance(tmpl_raw, str) else ""
        unsafe, safety_reasons = is_unsafe_chat_template(tmpl)
        if unsafe:
            logger.warning(
                "[PromptContract] Unsafe GGUF template detected → falling back to ChatML; reasons=%s",
                safety_reasons,
            )
            for reason in safety_reasons:
                logger.warning("[TemplateSafety] Unsafe template detected: reason=%s", reason)
            chatml_ok = "chatml" in handlers
            c = PromptContract(
                mode="messages",
                chat_format="chatml",
                prompt=None,
                messages=msg_payload,
                stop=stops_for_format("chatml"),
                template_source="fallback_unsafe_gguf",
                confidence="medium",
            )
            assert_prompt_contract(c)
            return PromptContractResolution(
                contract=c,
                handler_available=chatml_ok,
                template_safety={"unsafe": True, "reasons": list(safety_reasons)},
            )
        c = PromptContract(
            mode="messages",
            chat_format="chat_template.default",
            prompt=None,
            messages=msg_payload,
            stop=stops_for_format("chat_template.default"),
            template_source="gguf",
            confidence="high",
        )
        assert_prompt_contract(c)
        return PromptContractResolution(
            contract=c,
            handler_available=True,
            template_safety={"unsafe": False, "reasons": []},
        )

    # Step 2: Known family overrides.
    family_format: Optional[str] = None
    family_source: TemplateSource = "override"
    family_conf: PromptConfidence = "medium"

    if "nemotron" in model_name or "nvidia" in model_name or "phi" in model_name:
        family_format = "chatml"
    elif "mistral" in model_name or "mixtral" in model_name:
        family_format = "mistral-instruct"
    elif "llama-3" in model_name or "llama 3" in model_name:
        family_format = "llama-3"
    elif "qwen" in model_name:
        family_format = "chatml"
    elif "llama" in model_name:
        family_format = "llama-2"
    elif "oss" in model_name or "gpt" in model_name:
        # Step 3: OSS/GPT new handling.
        family_format = "chatml"
        family_source = "fallback"
        family_conf = "medium"
    else:
        # Step 4: unknown safe fallback.
        family_format = "chatml"
        family_source = "fallback"
        family_conf = "low"

    if family_format == "llama-2" and ("oss" in model_name or "gpt" in model_name):
        # Hard safety: never silently place OSS/GPT into llama-2.
        family_format = "chatml"
        family_source = "fallback"
        family_conf = "medium"

    supported = _format_supported(family_format, handlers)
    warning = None
    conf = family_conf
    if not supported:
        # Safe final fallback is always ChatML with explicit low confidence.
        family_format = "chatml"
        family_source = "fallback"
        conf = "low"
        warning = "Template could not be resolved safely; using fallback (chatml)."

    c = PromptContract(
        mode="messages",
        chat_format=family_format,
        prompt=None,
        messages=msg_payload,
        stop=stops_for_format(family_format),
        template_source=family_source,
        confidence=conf,
    )
    assert_prompt_contract(c)
    return PromptContractResolution(
        contract=c,
        warning=warning,
        handler_available=supported,
        template_safety=None,
    )


def build_rendered_contract(llama: Any, messages: list[dict[str, Any]]) -> PromptContract:
    prompt_txt, _fmt_stop, _note = reconstruct_formatted_prompt(llama, messages)
    c = PromptContract(
        mode="rendered",
        chat_format=str(getattr(llama, "chat_format", "") or None),
        prompt=prompt_txt or "",
        messages=None,
        stop=stops_for_format(getattr(llama, "chat_format", None)),
        template_source="fallback",
        confidence="low",
    )
    assert_prompt_contract(c)
    return c
