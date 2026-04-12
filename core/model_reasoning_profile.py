"""
Model capability classification: reasoning / thinking tokens vs plain chat (metadata only).

Does not change prompts, sampling, or outputs — detection + policy fields for routing/UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

ExecutionMode = Literal["direct", "thinking", "hybrid", "unknown"]

# Substrings matched against per-token vocab text and chat_template metadata.
THINKING_MARKERS: tuple[str, ...] = (
    "<redacted_thinking>",
    "<thinking>",
    "</redacted_thinking>",
    "</thinking>",
    "</think>",  # Qwen3 short think close tag
    "<|think|>",
    "<|thought|>",
)

NAME_REASONING_HINTS: tuple[str, ...] = (
    "deepseek",
    "r1",
    "reasoner",
    "reasoning",
    "thinking",
    "qwq",
    "qwen3",
    "qwen2.5",
    "magistral",
    "nemotron",
    "gpt-oss",
    "longcat",
    "minimax",
    "o1",
    "o3",
    "glm-4",
    "glm4",
)


@dataclass
class ModelReasoningProfile:
    model_name: str
    supports_thinking_tokens: bool
    thinking_token_patterns: List[str]
    default_mode: ExecutionMode
    reasoning_confidence: float  # 0–1
    detection_method: str  # tokenizer_scan | metadata | metadata_template | heuristic | hybrid_inference | fallback


def _basename(path: str | None) -> str:
    if not path:
        return ""
    return os.path.basename(path) or path


def _metadata_model_name(md: dict[str, Any]) -> str:
    for k in ("general.name", "general.basename", "name"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _llama_cpp_model(llama: Any) -> Any:
    """llama-cpp-python exposes the low-level model as ``_model`` (older) or ``model``."""
    mod = getattr(llama, "_model", None)
    if mod is not None:
        return mod
    return getattr(llama, "model", None)


def _scan_vocab_for_patterns(llama: Any, patterns: tuple[str, ...]) -> list[str]:
    """Return which marker substrings appear in any vocabulary piece text."""
    mod = _llama_cpp_model(llama)
    if mod is None:
        return []
    try:
        n = int(llama.n_vocab())
    except Exception:
        return []
    found: list[str] = []
    # Bound work for large vocabs; thinking tokens are usually early or mid-table
    cap = min(n, 262144)
    for tid in range(cap):
        try:
            txt = mod.token_get_text(tid)
        except Exception:
            continue
        if not txt:
            continue
        if isinstance(txt, bytes):
            try:
                s = txt.decode("utf-8", errors="ignore")
            except Exception:
                s = str(txt)
        else:
            s = str(txt)
        for p in patterns:
            if p in s and p not in found:
                found.append(p)
        # One matching piece is enough to classify; scanning until all markers hit is wasteful
        # and misses models that only expose a subset in the vocab.
        if found:
            break
    return found


def _metadata_template_signals(md: dict[str, Any]) -> list[str]:
    found: list[str] = []
    tmpl = md.get("tokenizer.chat_template")
    if isinstance(tmpl, str):
        low = tmpl.lower()
        for p in THINKING_MARKERS:
            if p.lower() in low or p in tmpl:
                found.append(p)
    return list(dict.fromkeys(found))


# Weaker signals: Jinja templates that mention reasoning/thinking without exact tag strings above.
_TEMPLATE_REASONING_HINTS: tuple[str, ...] = (
    "redacted_thinking",
    "reasoning_content",
    "</think>",
    "`<think>`",
    "internal reasoning",
    "chain of thought",
)


def _metadata_template_weak_signals(md: dict[str, Any]) -> list[str]:
    tmpl = md.get("tokenizer.chat_template")
    if not isinstance(tmpl, str) or not tmpl.strip():
        return []
    low = tmpl.lower()
    hits: list[str] = []
    for h in _TEMPLATE_REASONING_HINTS:
        if h in low:
            hits.append(h)
    return list(dict.fromkeys(hits))


def _name_heuristic(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in NAME_REASONING_HINTS)


def detect_model_reasoning_profile(
    llama: object,
    model_path: str | None = None,
) -> ModelReasoningProfile:
    """
    Classify model reasoning capability. Priority: tokenizer scan > metadata > name heuristic > fallback.
    """
    md: dict[str, Any] = {}
    if llama is not None:
        raw = getattr(llama, "metadata", None)
        if isinstance(raw, dict):
            md = raw

    display_name = _metadata_model_name(md) or _basename(model_path) or "unknown"
    vocab_hits = _scan_vocab_for_patterns(llama, THINKING_MARKERS) if llama is not None else []
    meta_hits = _metadata_template_signals(md)
    meta_weak = _metadata_template_weak_signals(md)
    path_name = _basename(model_path)
    name_hit = _name_heuristic(display_name) or _name_heuristic(path_name)

    thinking_token_patterns: list[str] = []
    detection_method = "unknown"
    confidence = 0.3
    supports = False

    if vocab_hits:
        thinking_token_patterns = list(vocab_hits)
        supports = True
        detection_method = "tokenizer_scan"
        confidence = 1.0
    elif meta_hits:
        thinking_token_patterns = list(meta_hits)
        supports = True
        detection_method = "metadata"
        confidence = 0.7
    elif meta_weak:
        thinking_token_patterns = list(meta_weak)
        supports = True
        detection_method = "metadata_template"
        confidence = 0.62
    elif name_hit:
        thinking_token_patterns = list(THINKING_MARKERS[:2])
        supports = True
        detection_method = "heuristic"
        confidence = 0.6

    default_mode: ExecutionMode = "direct"
    if not supports:
        default_mode = "direct"
        detection_method = "fallback"
        confidence = 0.3
    elif detection_method == "heuristic":
        # Name-only signal — treat as hybrid until tokenizer proves thinking tokens
        default_mode = "hybrid"
        detection_method = "hybrid_inference"
    else:
        default_mode = "thinking"

    return ModelReasoningProfile(
        model_name=display_name,
        supports_thinking_tokens=supports,
        thinking_token_patterns=thinking_token_patterns,
        default_mode=default_mode,
        reasoning_confidence=round(min(1.0, max(0.0, confidence)), 4),
        detection_method=detection_method,
    )


def profile_summary_dict(profile: ModelReasoningProfile) -> dict[str, Any]:
    return {
        "model_name": profile.model_name,
        "supports_thinking_tokens": profile.supports_thinking_tokens,
        "thinking_token_patterns": profile.thinking_token_patterns,
        "default_mode": profile.default_mode,
        "reasoning_confidence": profile.reasoning_confidence,
        "detection_method": profile.detection_method,
    }
