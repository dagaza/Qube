"""
High-ROI assistant-answer pattern extraction for referent promotion.

Pure functions; no Qt, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from core.discourse_state import TopicType

PatternId = Literal[
    "capital_of",
    "capital_of_alt",
    "founded_by",
    "founded_by_passive",
    "ceo_of",
    "president_of",
    "located_in",
    "superlative",
    "born_in",
]

_NAME = r"([A-Z][a-zA-Z''\u2019-]+(?:\s+(?:the|of|and|a|in)\s+[A-Za-z][a-zA-Z''\u2019-]+|\s+[A-Z][a-zA-Z''\u2019-]+)*)"
_PERSON = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
_ORG = _NAME

_PATTERNS: tuple[tuple[PatternId, re.Pattern[str], str, TopicType, float], ...] = (
    (
        "capital_of",
        re.compile(rf"^{_NAME}\s+is\s+the\s+capital\s+of\s+(.+?)[.!?]?\s*$", re.I),
        "1",
        "city",
        0.88,
    ),
    (
        "capital_of_alt",
        re.compile(rf"^the\s+capital\s+of\s+(.+?)\s+is\s+{_NAME}[.!?]?\s*$", re.I),
        "2",
        "city",
        0.88,
    ),
    (
        "founded_by",
        re.compile(rf"^{_PERSON}\s+founded\s+{_ORG}[.!?]?\s*$", re.I),
        "1",
        "person",
        0.82,
    ),
    (
        "founded_by_passive",
        re.compile(rf"^{_ORG}\s+was\s+founded\s+by\s+{_PERSON}[.!?]?\s*$", re.I),
        "2",
        "person",
        0.82,
    ),
    (
        "ceo_of",
        re.compile(
            rf"^{_PERSON}\s+is\s+(?:the\s+)?(?:CEO|chief executive)\s+of\s+{_ORG}[.!?]?\s*$",
            re.I,
        ),
        "1",
        "person",
        0.84,
    ),
    (
        "president_of",
        re.compile(rf"^{_PERSON}\s+is\s+(?:the\s+)?president\s+of\s+(.+?)[.!?]?\s*$", re.I),
        "1",
        "person",
        0.82,
    ),
    (
        "located_in",
        re.compile(rf"^{_NAME}\s+is\s+(?:located\s+)?in\s+(.+?)[.!?]?\s*$", re.I),
        "1",
        "city",
        0.80,
    ),
    (
        "superlative",
        re.compile(
            rf"^{_NAME}\s+is\s+the\s+(?:tallest|largest|highest|biggest|longest|smallest|deepest)\b",
            re.I,
        ),
        "1",
        "entity",
        0.78,
    ),
    (
        "born_in",
        re.compile(rf"^{_PERSON}\s+was\s+born\s+in\s+(.+?)[.!?]?\s*$", re.I),
        "1",
        "person",
        0.82,
    ),
)


@dataclass(frozen=True)
class AnswerPatternMatch:
    referent: str
    referent_type: TopicType
    pattern_id: str
    confidence: float
    query_entity: Optional[str] = None


def _first_sentence(text: str) -> str:
    return re.split(r"[.!?\n]", (text or "").strip(), maxsplit=1)[0].strip()


def extract_referent_from_assistant_answer(
    assistant_text: str,
    *,
    user_prompt: str = "",
) -> Optional[AnswerPatternMatch]:
    """Extract focal referent from a structured assistant answer."""
    first = _first_sentence(assistant_text)
    if not first:
        return None

    for pattern_id, pat, group, rtype, conf in _PATTERNS:
        m = pat.match(first)
        if not m:
            continue
        referent = (m.group(int(group)) or "").strip()[:120]
        if not referent:
            continue
        query_entity: Optional[str] = None
        if pattern_id == "capital_of":
            query_entity = (m.group(2) or "").strip()[:120] or None
        elif pattern_id == "capital_of_alt":
            query_entity = (m.group(1) or "").strip()[:120] or None
        return AnswerPatternMatch(
            referent=referent,
            referent_type=rtype,
            pattern_id=pattern_id,
            confidence=conf,
            query_entity=query_entity or None,
        )
    return None
