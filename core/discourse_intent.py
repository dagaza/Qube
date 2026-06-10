"""
Follow-up / discourse-continuation classification for chat turns.

Pure functions; no Qt, no LanceDB, no I/O.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.discourse_patterns import DEICTIC_PRONOUN_RE, has_possessive_anaphor

FOLLOW_UP_SUPPRESS_THRESHOLD = 0.55
FOLLOW_UP_SHORT_TOKEN_MAX = 14

_DISCOURSE_DEBUG_ENV = "QUBE_DISCOURSE_DEBUG"
_DISCOURSE_PROMPT_HINT_ENV = "QUBE_DISCOURSE_PROMPT_HINT"


class FollowUpKind(str, Enum):
    NONE = "none"
    ANAPHORIC = "anaphoric"
    ELLIPSIS = "ellipsis"
    WHY_HOW = "why_how"
    COMPARE = "compare"
    EXPAND = "expand"
    TIPS_FOR_THIS = "tips_for_this"


_WHY_HOW = re.compile(r"^\s*(why|how)\b", re.I)
_COMPARE = re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.I)
_EXPAND = re.compile(
    r"\b(expand on|elaborate|more detail|tell me more|go deeper|can you explain)\b",
    re.I,
)
_TIPS = re.compile(r"\btips?\b|\btricks?\b", re.I)
_WHAT_ABOUT = re.compile(r"\bwhat about\b", re.I)
_TOPIC_CHANGE = re.compile(
    r"\b(let'?s talk about|switch(?:ing)? to|change topic to|new topic)\b",
    re.I,
)

_DISCOURSE_TOPIC_SUFFIX_TEMPLATE = (
    " Active conversation topic: {topic}{type_hint}. "
    "Interpret 'this', 'that', and similar follow-ups accordingly."
)

_REFERENT_SALIENCE_PREFIX = (
    " Conversation context:\n"
    "Primary referent: {referent}{type_hint}.\n\n"
    "Resolve follow-up references (\"this\", \"that\", \"it\", \"they\", etc.) "
    "to the most relevant subject established in the conversation unless "
    "the user introduces a new one."
)


@dataclass(frozen=True)
class FollowUpClassification:
    kind: FollowUpKind
    confidence: float
    signals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def active(self) -> bool:
        return self.kind != FollowUpKind.NONE and self.confidence >= FOLLOW_UP_SUPPRESS_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "follow_up_kind": self.kind.value,
            "follow_up_confidence": round(self.confidence, 3),
            "follow_up_signals": list(self.signals),
        }


def discourse_debug_enabled() -> bool:
    return os.environ.get(_DISCOURSE_DEBUG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def discourse_prompt_hint_enabled() -> bool:
    return os.environ.get(_DISCOURSE_PROMPT_HINT_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def build_topic_salience_suffix(
    topic: str,
    *,
    topic_type: str = "unknown",
    max_chars: int = 120,
) -> str:
    """Short system suffix anchoring active topic near generation."""
    t = (topic or "").strip()
    if not t:
        return ""
    hint = ""
    if topic_type and topic_type not in ("unknown", ""):
        hint = f" ({topic_type})"
    text = _DISCOURSE_TOPIC_SUFFIX_TEMPLATE.format(topic=t[:60], type_hint=hint)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "."
    return text


def build_referent_salience_suffix(
    referent: str,
    *,
    referent_type: str = "unknown",
    max_chars: int = 320,
) -> str:
    """System suffix anchoring a resolved conversation referent for anaphoric follow-ups."""
    r = (referent or "").strip()
    if not r:
        return ""
    hint = ""
    if referent_type and referent_type not in ("unknown", ""):
        hint = f" ({referent_type})"
    text = _REFERENT_SALIENCE_PREFIX.format(referent=r[:60], type_hint=hint)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "."
    return text


def build_minimal_referent_fallback_suffix(
    referent: str,
    *,
    token: str = "it",
    max_chars: int = 80,
) -> str:
    """Short fallback hint when query rewrite did not succeed."""
    r = (referent or "").strip()
    t = (token or "it").strip()
    if not r:
        return ""
    text = f" Resolved reference: {t} → {r}."
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "."
    return text


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))


def _has_explicit_entity(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if re.search(r'"[^"]{2,}"|\'[^\']{2,}\'', s):
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", s):
        return True
    if re.search(r"\babout\s+[A-Za-z0-9]", s, re.I):
        return True
    return False


def _prior_turns(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not history:
        return []
    if len(history) >= 1 and str(history[-1].get("role", "")).lower() == "user":
        return list(history[:-1])
    return list(history)


def classify_follow_up(
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    discourse_state: Any | None = None,
) -> FollowUpClassification:
    """
    Classify whether the current turn is a discourse continuation.

    ``discourse_state`` may be a ``DiscourseState`` or any object with
    ``active_topic`` and optional ``confidence``.
    """
    text = (prompt or "").strip()
    if not text:
        return FollowUpClassification(FollowUpKind.NONE, 0.0)

    if _TOPIC_CHANGE.search(text):
        return FollowUpClassification(FollowUpKind.NONE, 0.0, ("topic_change",))

    prior = _prior_turns(history or [])
    if not prior:
        return FollowUpClassification(FollowUpKind.NONE, 0.0, ("no_prior_turns",))

    signals: list[str] = ["has_prior_turns"]
    kind = FollowUpKind.NONE
    score = 0.0

    deictic = bool(DEICTIC_PRONOUN_RE.search(text))
    possessive = has_possessive_anaphor(text)
    if _TIPS.search(text) and (deictic or possessive):
        kind = FollowUpKind.TIPS_FOR_THIS
        score = 0.72
        signals.append("tips+anaphoric")
    elif deictic or possessive:
        kind = FollowUpKind.ANAPHORIC
        score = 0.68 if possessive else 0.65
        signals.append("possessive_anaphor" if possessive else "anaphoric")
    elif _WHY_HOW.search(text):
        kind = FollowUpKind.WHY_HOW
        score = 0.62
        signals.append("why_how")
    elif _COMPARE.search(text):
        kind = FollowUpKind.COMPARE
        score = 0.60
        signals.append("compare")
    elif _EXPAND.search(text):
        kind = FollowUpKind.EXPAND
        score = 0.63
        signals.append("expand")
    elif _WHAT_ABOUT.search(text):
        kind = FollowUpKind.ANAPHORIC
        score = 0.58
        signals.append("what_about")

    tokens = _token_count(text)
    if tokens <= FOLLOW_UP_SHORT_TOKEN_MAX and not _has_explicit_entity(text):
        score += 0.08
        signals.append("short_query")
    elif _has_explicit_entity(text):
        score -= 0.25
        signals.append("explicit_entity_penalty")

    active_referent = (
        getattr(discourse_state, "active_referent", None) if discourse_state else None
    )
    active_topic = getattr(discourse_state, "active_topic", None) if discourse_state else None
    if (active_referent or active_topic) and kind != FollowUpKind.NONE:
        ds_conf = float(getattr(discourse_state, "confidence", 0.0) or 0.0)
        score += 0.12 + min(0.08, ds_conf * 0.08)
        if active_referent:
            score += 0.05
            signals.append("discourse_referent_boost")
        else:
            signals.append("discourse_topic_boost")

    if kind == FollowUpKind.NONE and tokens <= 6 and not _has_explicit_entity(text):
        kind = FollowUpKind.ELLIPSIS
        score = 0.50
        signals.append("ellipsis_short")

    score = max(0.0, min(1.0, score))
    if kind == FollowUpKind.NONE:
        score = min(score, 0.45)

    return FollowUpClassification(kind=kind, confidence=score, signals=tuple(signals))
