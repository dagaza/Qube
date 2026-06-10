"""
Chat formatting gate for task_type=chat (Phase 2).

Selects Harmony reply-shape guidance: brief, structured, or mixed.
Does not affect memory extraction, sidecar, stops, or task routing.
"""
from __future__ import annotations

import re
from typing import Literal

ChatFormatMode = Literal["brief", "structured", "mixed"]

_STRUCTURED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bexplain\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bhow\s+to\b",
        r"\bhow\s+do\s+i\b",
        r"\bwhy\s+(?:is|are|does|do|did|was|were)\b",
        r"\btroubleshoot\b",
        r"\bstep(?:s)?\s+to\b",
        r"\bwrite\s+(?:a|an|me)\b",
        r"\btell\s+me\s+a\s+(?:story|joke)\b",
        r"\bdescribe\b",
        r"\bsummarize\b",
        r"\bsummary\s+of\b",
        r"\blist\b",
        r"\bdifference\s+between\b",
        r"\bpros\s+and\s+cons\b",
        r"\bwalk\s+me\s+through\b",
        r"\binstructions?\s+for\b",
    )
)

_BRIEF_LOOKUP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"^what\s+is\b",
        r"^who\s+is\b",
        r"^when\s+is\b",
        r"^when\s+was\b",
        r"^where\s+is\b",
        r"^how\s+much\s+is\b",
        r"^how\s+many\b",
        r"\bcapital\s+of\b",
        r"\bconvert\b",
        r"\bdefinition\s+of\b",
        r"^define\b",
        r"^what\s+year\b",
        r"^what\s+date\b",
    )
)


def resolve_chat_format_mode(
    *,
    execution_route: str,
    user_query: str,
) -> ChatFormatMode:
    """
    Deterministic chat-only formatting mode from route + user query.

    Precedence: structured signals > retrieval routes > brief lookup > NONE default > mixed.
    """
    route = str(execution_route or "").upper().strip()
    q = (user_query or "").strip()
    if not q:
        return "mixed"

    q_lower = q.lower()

    for pat in _STRUCTURED_PATTERNS:
        if pat.search(q_lower):
            return "structured"

    if route in ("RAG", "HYBRID", "MEMORY", "WEB", "INTERNET"):
        return "structured"

    for pat in _BRIEF_LOOKUP_PATTERNS:
        if pat.search(q_lower):
            return "brief"

    if route == "NONE":
        return "brief"

    return "mixed"
