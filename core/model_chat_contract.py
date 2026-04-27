"""
Load-time binding of a single deterministic llama.cpp chat_format for a loaded model.

Does not replace PromptContract resolution; the native engine enforces this binding
after resolve_prompt_contract for messages-mode turns only.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

_VIOLATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"<\|channel\|", re.I), "<|channel|"),
    (re.compile(r"<\|start\|", re.I), "<|start|"),
    (re.compile(r"<\|start_header_id\|>", re.I), "<|start_header_id|>"),
    (re.compile(r"<\|im_start\|>assistant", re.I), "assistant_im_start_leak"),
    (re.compile(r"\banalysis\s*:", re.I), "analysis_block"),
    (re.compile(r"^\s*User\s*:", re.I | re.M), "role_reconstruction_user"),
    (re.compile(r"^\s*Assistant\s*:", re.I | re.M), "role_reconstruction_assistant"),
)


@dataclass
class ChatContract:
    model_name: str
    format_name: str
    tokenizer_template: Optional[str]
    source: str  # gguf | hf_config | manual_override | fallback
    locked: bool = True
    binding_reasoning: list[str] = field(default_factory=list)


@dataclass
class ChatContractResolution:
    model_name: str
    selected_format: str
    reasoning: list[str]
    fallback_used: bool


def build_model_info_from_llama(*, llama: Any, model_path: str) -> dict[str, Any]:
    md = getattr(llama, "metadata", None) or {}
    if not isinstance(md, dict):
        md = {}
    handlers = getattr(llama, "_chat_handlers", None) or {}
    keys: list[str] = []
    if isinstance(handlers, dict):
        keys = [str(k) for k in handlers.keys()]
    tmpl = md.get("tokenizer.chat_template")
    tmpl_s: Optional[str] = None
    if isinstance(tmpl, str) and tmpl.strip():
        tmpl_s = tmpl.strip()
    basename = os.path.basename(str(model_path or "").strip()) or "unknown"
    disp = ""
    for k in ("general.name", "general.basename", "name"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            disp = v.strip()
            break
    if not disp:
        disp = basename
    return {
        "model_path": str(model_path or ""),
        "model_basename": basename,
        "model_display_name": disp,
        "metadata": dict(md),
        "tokenizer_chat_template": tmpl_s,
        "chat_handler_keys": keys,
    }


def _handlers_set(model_info: dict[str, Any]) -> set[str]:
    raw = model_info.get("chat_handler_keys") or []
    return {str(x) for x in raw if str(x).strip()}


def _model_key(model_info: dict[str, Any]) -> str:
    n = str(model_info.get("model_display_name") or model_info.get("model_basename") or "").lower()
    return n


def resolve_chat_contract(model_info: dict[str, Any]) -> ChatContract:
    """
    Strict ordered resolution (no ML, no per-request calls).

    1) GGUF tokenizer.chat_template + handler chat_template.default
    2) Known family substrings
    3) Safe fallback: chatml
    """
    md = model_info.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}
    handlers = _handlers_set(model_info)
    tmpl = model_info.get("tokenizer_chat_template")
    if isinstance(tmpl, str) and tmpl.strip() and "chat_template.default" in handlers:
        name = str(model_info.get("model_display_name") or model_info.get("model_basename") or "unknown")
        return ChatContract(
            model_name=name,
            format_name="chat_template.default",
            tokenizer_template=tmpl.strip(),
            source="gguf",
            locked=True,
            binding_reasoning=["tokenizer.chat_template present and chat_template.default handler available"],
        )

    n = _model_key(model_info)
    reasoning = ["matched known model family"]
    fmt: str = "chatml"
    src = "hf_config"

    # Order aligned with prompt_contract family logic; OSS/GPT before generic "llama".
    if "nemotron" in n or "nvidia" in n or "phi" in n:
        fmt = "chatml"
    elif "mistral" in n or "mixtral" in n:
        fmt = "mistral-instruct"
    elif "llama-3" in n or "llama 3" in n or "llama3" in n:
        fmt = "llama-3"
    elif "oss" in n or "gpt" in n:
        fmt = "chatml"
        reasoning.append("oss_or_gpt_style_family -> chatml")
    elif "llama" in n or "meta-llama" in n:
        fmt = "llama-2"
    elif "qwen" in n:
        fmt = "chatml"
    else:
        fmt = "chatml"
        src = "fallback"
        reasoning = ["unknown_model_family -> chatml (single safe fallback)"]

    if fmt == "llama-2" and ("oss" in n or "gpt" in n):
        fmt, src = "chatml", "fallback"
        reasoning.append("hard_guard: never llama-2 for oss/gpt style name")

    if handlers and fmt not in handlers and fmt != "chat_template.default":
        reasoning.append(f"family_format {fmt} not in handlers; will map step may downgrade to chatml")
    name = str(model_info.get("model_display_name") or model_info.get("model_basename") or "unknown")
    return ChatContract(
        model_name=name,
        format_name=fmt,
        tokenizer_template=model_info.get("tokenizer_chat_template") if isinstance(model_info.get("tokenizer_chat_template"), str) else None,
        source=src,
        locked=True,
        binding_reasoning=reasoning,
    )


def map_chat_contract_to_llama_chat_format(
    chat: ChatContract,
    *,
    handler_keys: Optional[set[str]] = None,
) -> str:
    """Map bound contract to llama.cpp chat_format string; deterministic downgrade if unsupported."""
    fn = str(chat.format_name or "").strip()
    if not fn:
        return "chatml"
    if not handler_keys:
        return fn
    if fn in handler_keys:
        return fn
    if fn == "chat_template.default" and fn not in handler_keys:
        return "chatml" if "chatml" in handler_keys else fn
    if fn not in handler_keys and "chatml" in handler_keys:
        return "chatml"
    return fn


def chat_contract_to_resolution(chat: ChatContract) -> ChatContractResolution:
    return ChatContractResolution(
        model_name=chat.model_name,
        selected_format=chat.format_name,
        reasoning=list(chat.binding_reasoning),
        fallback_used=chat.source == "fallback",
    )


def detect_chat_contract_violation(text: str) -> tuple[bool, list[str]]:
    """Heuristic leakage markers (deterministic)."""
    if not (text or "").strip():
        return False, []
    hits: list[str] = []
    for pat, label in _VIOLATION_PATTERNS:
        if pat.search(text):
            hits.append(label)
    return bool(hits), hits
