"""
Shared deictic / possessive patterns for discourse tracking.

Single source of truth for discourse_state and discourse_intent.
"""
from __future__ import annotations

import re

DEICTIC_PRONOUN_RE = re.compile(
    r"\b("
    r"this|that|it|its|they|them|their|those|these|"
    r"he|him|his|she|her|hers|"
    r"we|our|us|you|your|"
    r"the same|the above|the latter"
    r")\b",
    re.I,
)

_POSSESSIVE_ANAPHOR_RE = re.compile(
    r"\b(its|his|her|their|our|your)\s+\w+",
    re.I,
)


def is_deictic_prompt(text: str) -> bool:
    """True when the prompt contains deictic follow-up markers."""
    return bool(DEICTIC_PRONOUN_RE.search((text or "").strip()))


def is_deictic_topic_phrase(topic: str) -> bool:
    """True when an extracted topic phrase still contains unresolved deictic refs."""
    t = (topic or "").strip()
    if not t:
        return False
    return bool(DEICTIC_PRONOUN_RE.search(t)) or bool(_POSSESSIVE_ANAPHOR_RE.search(t))


def has_possessive_anaphor(text: str) -> bool:
    """True when text contains a possessive pronoun anchoring a noun phrase."""
    return bool(_POSSESSIVE_ANAPHOR_RE.search((text or "").strip()))
