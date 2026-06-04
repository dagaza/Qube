"""
Harmony protocol detection and contract metadata for gpt-oss / Harmony GGUF families.

Used by prompt contract resolution, template safety, streaming parsers, and validators.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

HarmonyDetectionMethod = Literal["name", "metadata", "template", "name+template"]

HARMONY_FINAL_ANCHOR = "<|start|>assistant<|channel|>final<|message|>"

# Primary Harmony EOS — phrase stops are emergency-only (see harmony_phrase_stops_enabled).
HARMONY_PRIMARY_STOPS: tuple[str, ...] = ("<|return|>",)

HARMONY_EMERGENCY_PHRASE_STOPS: tuple[str, ...] = (
    "\nWe need to",
    " We need to",
    "\nWe should",
    " We should",
    "\nWe have",
    " We have",
    "\nWe have to",
    " We have to",
    "\nLet's",
    " Let's",
    "\nLet's clarify",
    "Let's clarify",
    "\nThe user says",
    "The user says",
    "\nThe question says",
    "The question says",
    "\nThe user wants",
    " The user wants",
    "\nThey may be asking",
    " They may be asking",
    "\nno meta commentary",
    " no meta commentary",
)

_HARMONY_TEMPLATE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|start\|>", re.I),
    re.compile(r"<\|channel\|>", re.I),
    re.compile(r"<\|message\|>", re.I),
    re.compile(r"<\|end\|>", re.I),
)

_ARCH_KEYS = (
    "general.architecture",
    "general.file_type",
    "tokenizer.ggml.model",
)


@dataclass(frozen=True)
class HarmonyProtocol:
    """Resolved Harmony profile for the loaded model."""

    model_name: str
    detection_method: HarmonyDetectionMethod


def is_harmony_model_name(model_name: str) -> bool:
    n = (model_name or "").lower()
    if "gpt-oss" in n or "gptoss" in n:
        return True
    if "gpt" in n and "oss" in n:
        return True
    return False


def template_indicates_harmony(chat_template: str) -> bool:
    if not isinstance(chat_template, str) or not chat_template.strip():
        return False
    hits = sum(1 for pat in _HARMONY_TEMPLATE_MARKERS if pat.search(chat_template))
    return hits >= 3


def metadata_indicates_harmony(metadata: dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return False
    for key in _ARCH_KEYS:
        val = metadata.get(key)
        if isinstance(val, str) and val.strip():
            low = val.lower()
            if "gptoss" in low or "gpt-oss" in low or "gpt_oss" in low:
                return True
    for k in ("general.name", "general.basename"):
        v = metadata.get(k)
        if isinstance(v, str) and is_harmony_model_name(v):
            return True
    return False


def detect_harmony_protocol(
    *,
    model_name: str,
    metadata: Optional[dict[str, Any]] = None,
    chat_template: Optional[str] = None,
) -> Optional[HarmonyProtocol]:
    """
    Return a Harmony profile when the model is a known Harmony family member.

    Prefers metadata/name; template markers alone are a weaker signal (still accepted).
    """
    md = metadata if isinstance(metadata, dict) else {}
    tmpl = chat_template if isinstance(chat_template, str) else md.get("tokenizer.chat_template")
    tmpl_str = tmpl if isinstance(tmpl, str) else ""

    name_hit = is_harmony_model_name(model_name)
    meta_hit = metadata_indicates_harmony(md)
    tmpl_hit = template_indicates_harmony(tmpl_str)

    if not (name_hit or meta_hit or tmpl_hit):
        return None

    if name_hit and tmpl_hit:
        method: HarmonyDetectionMethod = "name+template"
    elif name_hit or meta_hit:
        method = "name" if name_hit else "metadata"
    else:
        method = "template"

    display = (model_name or "").strip() or "unknown"
    return HarmonyProtocol(model_name=display, detection_method=method)


def harmony_stops_for_contract(*, include_phrase_stops: bool | None = None) -> list[str]:
    """Merged stop list for Harmony rendered completions."""
    if include_phrase_stops is None:
        include_phrase_stops = not harmony_phrase_stops_disabled()
    stops = list(HARMONY_PRIMARY_STOPS)
    if include_phrase_stops:
        stops.extend(HARMONY_EMERGENCY_PHRASE_STOPS)
    return stops


def harmony_phrase_stops_disabled() -> bool:
    """Disable English planning phrase stops (expert rollback). Default: phrase stops on."""
    return os.environ.get("QUBE_HARMONY_PHRASE_STOPS", "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )


def harmony_phrase_stops_enabled() -> bool:
    return not harmony_phrase_stops_disabled()


def harmony_stream_parser_enabled() -> bool:
    """Streaming Harmony parser (default on). Set QUBE_HARMONY_PARSER=0 to disable."""
    v = os.environ.get("QUBE_HARMONY_PARSER", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def is_harmony_contract(contract: Any | None) -> bool:
    if contract is None:
        return False
    if getattr(contract, "protocol", None) == "harmony":
        return True
    prompt = (contract.prompt or "").strip()
    if contract.mode == "rendered" and prompt.endswith(HARMONY_FINAL_ANCHOR):
        return True
    return False


def is_expected_harmony_chat_template(template: str) -> bool:
    """Harmony GGUF templates are expected to contain protocol markers."""
    return template_indicates_harmony(template)
