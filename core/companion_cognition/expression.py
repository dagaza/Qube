"""Expression renderers L0/L1 and ExpressionRouter."""

from __future__ import annotations

import os
import re

from core.companion_cognition.ambient_context import AmbientContext
from core.companion_cognition.message_library import (
    CuratedMessage,
    MessageLibrary,
    MessageTemplate,
    get_message_library,
)
from core.companion_cognition.personality import CompanionPersonalityVector
from core.companion_cognition.thoughts import kind_for_intent
from core.companion_cognition.types import (
    CompanionObservation,
    CompanionThought,
    ExpressionLevel,
    ExpressionPlan,
    ExpressionResult,
)
from core.companion_cognition.variety import VarietySnapshot, get_variety_store
from core.companion_line_quality import is_acceptable_companion_line
from core.companion_verbal_prompts import COMPANION_LINE_MAX_CHARS, truncate_companion_caption

_BASENAME_SLOT_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def sanitize_slot_value(key: str, value: object) -> str:
    if key == "basename":
        raw = str(value or "").strip()
        base = os.path.basename(raw.replace("\\", "/"))
        if _BASENAME_SLOT_RE.match(base):
            return base
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "", base)[:80]
        return cleaned
    if key == "file_count_word":
        return str(value or "a few")[:20]
    if key == "milestone_id":
        return re.sub(r"[^a-z0-9_]", "", str(value or "").lower())[:32]
    return str(value or "")[:40]


def safe_slots(thought: CompanionThought) -> dict[str, str]:
    return {k: sanitize_slot_value(k, v) for k, v in (thought.slots or {}).items()}


def render_template(tpl: MessageTemplate, slots: dict[str, str]) -> str:
    allowed = set(tpl.placeholders)
    fmt = {k: v for k, v in slots.items() if k in allowed}
    try:
        line = tpl.pattern.format(**fmt)
    except KeyError:
        return ""
    return truncate_companion_caption(line, COMPANION_LINE_MAX_CHARS)


def render_l0(msg: CuratedMessage) -> str:
    return truncate_companion_caption(msg.text, COMPANION_LINE_MAX_CHARS)


def try_local_expression(
    thought: CompanionThought,
    obs: CompanionObservation,
    trigger: str,
    variety: VarietySnapshot,
    personality: CompanionPersonalityVector,
    *,
    library: MessageLibrary | None = None,
    ambient: AmbientContext | None = None,
) -> ExpressionResult | None:
    lib = library or get_message_library()
    store = get_variety_store()
    for_preview = obs.type == "settings_preview"

    msg = lib.select_message(
        thought, variety, personality, ambient, for_preview=for_preview
    )
    if msg is not None:
        line = render_l0(msg)
        dup_ok = for_preview or not store.is_semantic_duplicate(line)
        if line and is_acceptable_companion_line(line) and dup_ok:
            return ExpressionResult(
                line=line,
                kind=kind_for_intent(thought.intent, obs.type),
                trigger=trigger,
                level=ExpressionLevel.CURATED,
                message_id=msg.id,
                intent=thought.intent,
                mood=thought.mood,
                voice=msg.voice,
                motifs=msg.motifs,
            )

    if thought.slots:
        tpl = lib.select_template(thought, variety, personality)
        if tpl is not None:
            line = render_template(tpl, safe_slots(thought))
            if line and is_acceptable_companion_line(line) and not store.is_semantic_duplicate(line):
                return ExpressionResult(
                    line=line,
                    kind=kind_for_intent(thought.intent, obs.type),
                    trigger=trigger,
                    level=ExpressionLevel.TEMPLATE,
                    message_id=tpl.id,
                    intent=thought.intent,
                    mood=thought.mood,
                    voice=tpl.voice,
                )
    return None


def build_expression_plan(
    thought: CompanionThought,
    obs: CompanionObservation,
    trigger: str,
    variety: VarietySnapshot,
    personality: CompanionPersonalityVector,
    max_level: ExpressionLevel,
    *,
    library: MessageLibrary | None = None,
    ambient: AmbientContext | None = None,
) -> ExpressionPlan | None:
    """Plan local or sidecar expression up to max_level."""
    lib = library or get_message_library()
    kind = kind_for_intent(thought.intent, obs.type)

    if max_level >= ExpressionLevel.CURATED:
        msg = lib.select_message(thought, variety, personality, ambient)
        if msg is not None:
            return ExpressionPlan(
                level=ExpressionLevel.CURATED,
                message_id=msg.id,
                seed_line=render_l0(msg),
                kind=kind,
            )

    if max_level >= ExpressionLevel.TEMPLATE and thought.slots:
        tpl = lib.select_template(thought, variety, personality)
        if tpl is not None:
            line = render_template(tpl, safe_slots(thought))
            if line:
                return ExpressionPlan(
                    level=ExpressionLevel.TEMPLATE,
                    template_id=tpl.id,
                    message_id=tpl.id,
                    seed_line=line,
                    kind=kind,
                )

    if max_level >= ExpressionLevel.SIDECAR_REWRITE:
        msg = lib.select_message(thought, variety, personality, ambient)
        seed = render_l0(msg) if msg else ""
        if not seed:
            seed = _fallback_seed(thought)
        return ExpressionPlan(
            level=ExpressionLevel.SIDECAR_REWRITE,
            message_id=msg.id if msg else "",
            seed_line=seed,
            kind=kind,
        )

    if max_level >= ExpressionLevel.FULL_GENERATE:
        return ExpressionPlan(
            level=ExpressionLevel.FULL_GENERATE,
            seed_line="",
            kind=kind,
        )
    return None


def _fallback_seed(thought: CompanionThought) -> str:
    seeds = {
        "wellbeing": "Still here if you need me.",
        "atmosphere": "The room feels settled.",
        "celebration": "All set on my end.",
        "acknowledge_effort": "That's in your library now.",
        "curiosity": "Something interesting might be brewing.",
        "humor": "Current status: observing professionally.",
        "reflection": "Small steps are surprisingly persistent.",
        "self_expression": "I've decided this is a good observing spot.",
    }
    return seeds.get(thought.intent, "Still here if you need me.")
