"""Vocab/template-gated stop list filtering for native prompt bundles."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

from core.harmony_protocol import (
    HARMONY_EMERGENCY_PHRASE_STOPS,
    harmony_phrase_stops_disabled,
    is_harmony_model_name,
)
from core.model_reasoning_profile import THINKING_MARKERS, _llama_cpp_model
from core.native_llm_debug import llama_eos_bos_strings, llm_debug_enabled
from core.qwen3_thinking_policy import is_qwen3_model

logger = logging.getLogger("Qube.PromptTemplateRouter")
_debug_logger = logging.getLogger("Qube.NativeLLM.Debug")

StopKind = Literal[
    "protected",
    "control_token",
    "thinking_tag",
    "phrase_sentinel",
    "format_marker",
    "other",
]

_CONTROL_TOKEN_RE = re.compile(r"^</?\|[^|\n]{1,56}\|>$")
_FORMAT_MARKERS: frozenset[str] = frozenset({"</s>", "[INST]", "[/INST]", "<|end|>"})
_THINKING_TAG_STOPS: frozenset[str] = frozenset(
    {
        *THINKING_MARKERS,
        "<think>",
        "</think>",
        "<thinking>",
        "</thinking>",
    }
)
_QWEN_PHRASE_STOPS: frozenset[str] = frozenset(
    {
        "Thinking Process:",
        "\nThinking Process:",
        "1. **Analyze",
        "\n1. **Analyze",
    }
)
_PHRASE_SENTINEL_STOPS: frozenset[str] = frozenset(
    {*HARMONY_EMERGENCY_PHRASE_STOPS, *_QWEN_PHRASE_STOPS}
)
_NATIVE_THINKING_TEMPLATE_TYPES: frozenset[str] = frozenset({"chatml", "llama3", "phi"})
_JINJA_CHAT_FORMATS: frozenset[str] = frozenset({"chat_template.default", "jinja"})


@dataclass(frozen=True)
class DroppedStop:
    stop: str
    kind: StopKind
    reason: str


@dataclass
class StopFilterReport:
    kept: list[str] = field(default_factory=list)
    dropped: list[DroppedStop] = field(default_factory=list)

    @property
    def dropped_stops(self) -> list[str]:
        return [d.stop for d in self.dropped]


def _chat_template_text(llama: Any) -> str:
    md = getattr(llama, "metadata", None) or {}
    if not isinstance(md, dict):
        return ""
    tmpl = md.get("tokenizer.chat_template")
    return tmpl if isinstance(tmpl, str) else ""


def _collect_vocab_pieces(llama: Any) -> tuple[str, ...]:
    mod = _llama_cpp_model(llama)
    if mod is None:
        return ()
    try:
        n = int(llama.n_vocab())
    except Exception:
        return ()
    pieces: list[str] = []
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
        if s:
            pieces.append(s)
    return tuple(pieces)


def _stop_in_vocab_or_template(
    stop: str,
    *,
    chat_template: str,
    vocab_pieces: tuple[str, ...],
) -> bool:
    if not stop:
        return False
    if chat_template and stop in chat_template:
        return True
    return any(stop in piece for piece in vocab_pieces)


def _uses_jinja_gguf_template(
    effective_chat_format: Optional[str],
    llama: Any,
) -> bool:
    cf = (effective_chat_format or getattr(llama, "chat_format", "") or "").strip().lower()
    if cf in _JINJA_CHAT_FORMATS:
        return True
    return cf.startswith("chat_template.")


def _classify_stop(stop: str) -> StopKind:
    if stop in _THINKING_TAG_STOPS:
        return "thinking_tag"
    if stop in _PHRASE_SENTINEL_STOPS:
        return "phrase_sentinel"
    if stop in _FORMAT_MARKERS:
        return "format_marker"
    if _CONTROL_TOKEN_RE.match(stop):
        return "control_token"
    return "other"


def _normalize_protected(protected_stops: Sequence[str] | None, eos_token: str) -> frozenset[str]:
    out: set[str] = set()
    for item in protected_stops or ():
        s = str(item or "")
        if s:
            out.add(s)
    if eos_token:
        out.add(eos_token)
    return frozenset(out)


def _should_keep_stop(
    stop: str,
    *,
    kind: StopKind,
    protected: frozenset[str],
    chat_template: str,
    vocab_pieces: tuple[str, ...],
    template_type: str,
    model_name: str,
    model_path: str,
    jinja_runtime: bool,
) -> tuple[bool, str]:
    if stop in protected:
        return True, "protected"

    has_evidence = _stop_in_vocab_or_template(
        stop,
        chat_template=chat_template,
        vocab_pieces=vocab_pieces,
    )
    tt = (template_type or "fallback").lower()

    if kind == "control_token":
        if has_evidence:
            return True, "control_token_vocab_or_template"
        return False, "control_token_no_vocab_or_template_evidence"

    if kind == "thinking_tag":
        if jinja_runtime:
            if has_evidence:
                return True, "thinking_tag_jinja_with_vocab_evidence"
            return False, "thinking_tag_jinja_without_vocab_evidence"
        if tt in _NATIVE_THINKING_TEMPLATE_TYPES and has_evidence:
            return True, "thinking_tag_native_template_with_evidence"
        if tt in _NATIVE_THINKING_TEMPLATE_TYPES and not has_evidence:
            return False, "thinking_tag_native_template_without_evidence"
        return False, "thinking_tag_unsupported_template_family"

    if kind == "phrase_sentinel":
        if stop in _QWEN_PHRASE_STOPS:
            if is_qwen3_model(model_path=model_path, model_name=model_name):
                return True, "qwen_phrase_sentinel"
            return False, "qwen_phrase_sentinel_non_qwen_model"
        if harmony_phrase_stops_disabled():
            return False, "harmony_phrase_stops_disabled"
        if is_harmony_model_name(model_name):
            return True, "harmony_phrase_sentinel"
        return False, "phrase_sentinel_non_harmony_model"

    if kind == "format_marker":
        if stop == "</s>" and tt == "mistral":
            return True, "mistral_format_marker"
        if stop in ("[INST]", "[/INST]") and tt in ("mistral", "llama-2"):
            return True, "instruct_format_marker"
        if stop == "<|end|>" and tt == "phi":
            return True, "phi_format_marker"
        if has_evidence:
            return True, "format_marker_vocab_or_template"
        return False, "format_marker_no_evidence"

    if has_evidence:
        return True, "other_vocab_or_template_evidence"
    return False, "other_no_vocab_or_template_evidence"


def filter_stop_tokens(
    llama: Any,
    stops: Sequence[str],
    *,
    template_type: str,
    model_name: str = "",
    model_path: str = "",
    effective_chat_format: Optional[str] = None,
    protected_stops: Sequence[str] | None = None,
) -> tuple[list[str], StopFilterReport]:
    """
    Filter merged stop strings to those supported by the loaded tokenizer/template.

    Formatter/contract/EOS stops in ``protected_stops`` are always kept.
    """
    eos_token, _ = llama_eos_bos_strings(llama)
    protected = _normalize_protected(protected_stops, eos_token)
    chat_template = _chat_template_text(llama)
    jinja_runtime = _uses_jinja_gguf_template(effective_chat_format, llama)

    needs_vocab = False
    for stop in stops:
        s = str(stop or "")
        if not s or s in protected:
            continue
        needs_vocab = True
        break
    vocab_pieces = _collect_vocab_pieces(llama) if needs_vocab else ()

    report = StopFilterReport()
    seen: set[str] = set()
    for stop in stops:
        s = str(stop or "")
        if not s or s in seen:
            continue
        seen.add(s)
        kind = _classify_stop(s)
        keep, reason = _should_keep_stop(
            s,
            kind=kind,
            protected=protected,
            chat_template=chat_template,
            vocab_pieces=vocab_pieces,
            template_type=template_type,
            model_name=model_name,
            model_path=model_path,
            jinja_runtime=jinja_runtime,
        )
        if keep:
            report.kept.append(s)
        else:
            report.dropped.append(DroppedStop(stop=s, kind=kind, reason=reason))

    if report.dropped:
        logger.info(
            "[LLM-STOP-FILTER] kept=%d dropped=%d jinja_runtime=%s",
            len(report.kept),
            len(report.dropped),
            jinja_runtime,
        )
        for item in report.dropped:
            logger.info(
                "[LLM-STOP-FILTER] drop stop=%r kind=%s reason=%s",
                item.stop,
                item.kind,
                item.reason,
            )
    if llm_debug_enabled() and report.dropped:
        _debug_logger.info(
            json.dumps(
                {
                    "event": "stop_token_filter",
                    "template_type": template_type,
                    "model_name": model_name,
                    "jinja_runtime": jinja_runtime,
                    "kept_count": len(report.kept),
                    "dropped_count": len(report.dropped),
                    "kept": report.kept,
                    "dropped": [
                        {"stop": d.stop, "kind": d.kind, "reason": d.reason}
                        for d in report.dropped
                    ],
                },
                ensure_ascii=False,
            )
        )
    return report.kept, report


__all__ = [
    "DroppedStop",
    "StopFilterReport",
    "filter_stop_tokens",
]
