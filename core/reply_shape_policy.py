"""
Unified reply-shape policy: intent, formatting mode, and instruction coherence.

Combines route/query signals, discourse follow-up classification, and
instruction-conflict resolution into one deterministic policy per turn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from core.chat_format_mode import ChatFormatMode, resolve_chat_format_mode
from core.discourse_intent import FollowUpClassification, FollowUpKind

FormatIntent = Literal[
    "brief",
    "structured",
    "mixed",
    "enumeration",
    "follow_up",
]

_STRUCTURED_HINT = (
    " When the user asks for a list, comparison, or step-by-step explanation, "
    "use clear structure (bullets or numbered items as appropriate). "
    "Do not force a single sentence when more structure improves clarity."
)

_BRIEF_HINT = (
    " Keep the response concise — typically one short paragraph or a few sentences."
)

_ENUMERATION_HINT = (
    " The user is asking for an enumeration. Provide a complete, well-organized list. "
    "Do not truncate to a single concise sentence."
)

_FOLLOW_UP_HINT = (
    " This is a follow-up question; use prior conversation context to resolve "
    "pronouns and references. If uncertain about the referent, ask briefly rather "
    "than guessing."
)

_UNCERTAINTY_HINT = (
    " If you are not confident in a factual claim, say so rather than inventing details."
)

_ENUMERATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\blist\b",
        r"\bname\b.{0,36}\b(all|every|each|major|main|primary|key)\b",
        r"\bwhat are\b.{0,48}\b("
        r"groups|types|kinds|examples|reasons|steps|ways|sites|attractions|"
        r"languages|minorities|festivals|dishes|regions|categories|options"
        r")\b",
        r"\bwhich\b.{0,48}\b(are|were|include|contain|exist)\b",
        r"\bhow many\b",
        r"\bmajor\b.{0,36}\b("
        r"groups|ethnic|religious|languages|festivals|attractions|sites|rivers|peaks"
        r")\b",
        r"\btypes of\b",
        r"\bkinds of\b",
        r"\benumerate\b",
        r"\bname the\b",
    )
)


@dataclass(frozen=True)
class ReplyShapePolicy:
    chat_format_mode: ChatFormatMode
    format_intent: FormatIntent
    allow_structured_output: bool
    require_list_format: bool
    system_reply_hint: str
    instruction_conflicts: tuple[str, ...]
    resolution_notes: tuple[str, ...]

    def trace_fields(self) -> dict[str, Any]:
        return {
            "chat_format_mode": self.chat_format_mode,
            "format_intent": self.format_intent,
            "allow_structured_output": self.allow_structured_output,
            "require_list_format": self.require_list_format,
            "instruction_conflicts": list(self.instruction_conflicts),
            "resolution_notes": list(self.resolution_notes),
        }


def detect_enumeration_intent(user_query: str) -> bool:
    q = (user_query or "").strip()
    if not q:
        return False
    q_lower = q.lower()
    for pat in _ENUMERATION_PATTERNS:
        if pat.search(q_lower):
            return True
    return False


def _base_system_hint(
    *,
    format_intent: FormatIntent,
    prior_turn_unreliable: bool,
) -> str:
    parts: list[str] = []
    if format_intent == "enumeration":
        parts.append(_ENUMERATION_HINT)
    elif format_intent == "structured":
        parts.append(_STRUCTURED_HINT)
    elif format_intent == "follow_up":
        parts.append(_FOLLOW_UP_HINT)
    elif format_intent == "brief":
        parts.append(_BRIEF_HINT)
    if prior_turn_unreliable:
        parts.append(_UNCERTAINTY_HINT)
    return "".join(parts).strip()


def resolve_reply_shape_policy(
    *,
    execution_route: str,
    user_query: str,
    follow_up: FollowUpClassification | None = None,
    prior_turn_unreliable: bool = False,
    has_retrieval_sources: bool = False,
) -> ReplyShapePolicy:
    """
    Resolve chat formatting mode and non-Harmony system hints for one turn.

    Applies instruction-conflict resolution when brevity constraints would fight
    list/enumeration or compare intents.
    """
    base_mode = resolve_chat_format_mode(
        execution_route=execution_route,
        user_query=user_query,
    )
    conflicts: list[str] = []
    notes: list[str] = []
    require_list = detect_enumeration_intent(user_query)
    fu = follow_up or FollowUpClassification(FollowUpKind.NONE, 0.0)

    format_intent: FormatIntent
    if require_list:
        format_intent = "enumeration"
    elif fu.active and fu.kind in (
        FollowUpKind.ANAPHORIC,
        FollowUpKind.ELLIPSIS,
        FollowUpKind.TIPS_FOR_THIS,
    ):
        format_intent = "follow_up"
    elif fu.active and fu.kind in (FollowUpKind.COMPARE, FollowUpKind.EXPAND):
        format_intent = "structured"
    elif base_mode == "structured":
        format_intent = "structured"
    elif base_mode == "brief":
        format_intent = "brief"
    else:
        format_intent = "mixed"

    mode = base_mode

    if require_list and mode == "brief":
        conflicts.append("brief_vs_enumeration")
        mode = "structured"
        notes.append("upgraded_brief_to_structured_for_enumeration")

    if fu.active and fu.kind == FollowUpKind.COMPARE and mode == "brief":
        conflicts.append("brief_vs_compare")
        mode = "structured"
        notes.append("upgraded_brief_to_structured_for_compare")

    if format_intent == "follow_up" and mode == "brief" and not require_list:
        mode = "mixed"
        notes.append("relaxed_brief_to_mixed_for_follow_up")

    if has_retrieval_sources and mode == "brief" and require_list:
        mode = "structured"
        notes.append("structured_for_grounded_enumeration")

    allow_structured = mode in ("structured", "mixed") or require_list
    if require_list and not allow_structured:
        allow_structured = True
        notes.append("forced_structured_for_enumeration")

    hint = _base_system_hint(
        format_intent=format_intent,
        prior_turn_unreliable=prior_turn_unreliable,
    )

    return ReplyShapePolicy(
        chat_format_mode=mode,
        format_intent=format_intent,
        allow_structured_output=allow_structured,
        require_list_format=require_list,
        system_reply_hint=hint,
        instruction_conflicts=tuple(conflicts),
        resolution_notes=tuple(notes),
    )
