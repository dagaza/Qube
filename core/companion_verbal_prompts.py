"""Sidecar prompt assembly for companion commentary lines."""

from __future__ import annotations

import re
from typing import Any

from core.cognition_prompt_adapter import build_cognition_prompt
from core.companion_verbal_traits import (
    CompanionVerbalTraitPreset,
    normalize_companion_verbal_trait,
    trait_system_fragment,
)

COMPANION_LINE_MAX_CHARS = 72


def truncate_companion_caption(
    text: str,
    max_chars: int = COMPANION_LINE_MAX_CHARS,
    *,
    ellipsis: str = "…",
) -> str:
    """Trim to max length on a word boundary (never mid-word) with an ellipsis."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= len(ellipsis):
        return ellipsis[:max_chars]
    budget = max_chars - len(ellipsis)
    cut = cleaned[:budget]
    next_ch = cleaned[budget : budget + 1]
    if cut and next_ch and not next_ch.isspace() and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.rstrip(" \t.,;:!?")
    if not cut:
        cut = cleaned[:budget]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        cut = cut.rstrip(" \t.,;:!?")
    return f"{cut}{ellipsis}"


_VALID_KINDS = frozenset({"idle_quip", "ingest_ack", "download_ack", "skip"})
_VALID_TRIGGERS = frozenset({"idle", "ingest_complete", "download_complete", "test"})

_BASE_SYSTEM = (
    "You are the Qube desktop companion — not the main chat assistant. "
    "Write ONE short line the user reads under a small desktop orb — speak TO them, "
    "not about the task of writing a caption. "
    "Respond with STRICT JSON only — no other text. "
    'Format: {"line":"your line here","kind":"idle_quip|ingest_ack|download_ack|skip"}. '
    'Good: {"line":"Still here if you need me.","kind":"idle_quip"}. '
    'Bad: {"line":"Maybe something about the companion","kind":"idle_quip"}. '
    "Rules: line must be under 72 characters; no markdown; "
    "never mention JSON, captions, triggers, or 'the companion' as a concept; "
    "do not mention private user data, memories, or chat history; "
    "do not claim you performed work you did not do; "
    'if unsure or nothing fits, return {"line":"","kind":"skip"}.'
)


def build_companion_line_system(
    *,
    trait_preset: CompanionVerbalTraitPreset | str = CompanionVerbalTraitPreset.NEUTRAL,
    user_system_prompt: str = "",
) -> str:
    parts = [_BASE_SYSTEM, trait_system_fragment(trait_preset)]
    extra = (user_system_prompt or "").strip()
    if extra:
        parts.append(f"User companion style notes: {extra[:800]}")
    return "\n\n".join(parts)


def build_companion_line_user_payload(
    *,
    trigger: str,
    file_count: int | None = None,
    filename: str | None = None,
    basename: str | None = None,
) -> str:
    trig = str(trigger or "idle").strip().lower()
    if trig not in _VALID_TRIGGERS:
        trig = "idle"
    lines = [f"trigger: {trig}"]
    if file_count is not None and file_count > 0:
        lines.append(f"file_count: {int(file_count)}")
    if filename:
        lines.append(f"filename: {str(filename)[:120]}")
    if basename:
        lines.append(f"basename: {str(basename)[:120]}")
    if trig == "idle":
        lines.append("context: user is idle; companion is visible on desktop.")
        lines.append(
            "Write a casual line to the user at their desk (e.g. a gentle check-in). "
            "Do not talk about companions, captions, or JSON."
        )
    elif trig == "ingest_complete":
        lines.append("context: a document finished ingesting into the library.")
    elif trig == "download_complete":
        lines.append("context: a model file finished downloading.")
    elif trig == "test":
        lines.append(
            "context: settings preview only — output STRICT JSON with one short caption "
            "under 72 characters. Do not write welcome text, instructions, or explanations."
        )
        lines.append('required shape: {"line":"your caption here","kind":"idle_quip"}')
    return "\n".join(lines)


def build_companion_line_prompt(
    *,
    chat_format: str = "chatml",
    model_path: str = "",
    trait_preset: CompanionVerbalTraitPreset | str = CompanionVerbalTraitPreset.NEUTRAL,
    user_system_prompt: str = "",
    trigger: str = "idle",
    file_count: int | None = None,
    filename: str | None = None,
    basename: str | None = None,
    **kwargs: Any,
) -> str:
    _ = kwargs
    preset = normalize_companion_verbal_trait(trait_preset)
    system = build_companion_line_system(
        trait_preset=preset,
        user_system_prompt=user_system_prompt,
    )
    user = build_companion_line_user_payload(
        trigger=trigger,
        file_count=file_count,
        filename=filename,
        basename=basename,
    )
    return build_cognition_prompt(system, user, chat_format, model_path=model_path)
