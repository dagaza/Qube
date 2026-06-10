"""
Follow-up prompt grounding: validated anchors for user-visible prompt injection and salience.

Rejects meaningless fragments (numbers, measurements, dates) extracted from prior assistant text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from core.discourse_patterns import has_possessive_anaphor, is_deictic_prompt, is_deictic_topic_phrase
from core.discourse_query_rewrite import REWRITE_CONFIDENCE_MIN, ResolvedUserQuery

if TYPE_CHECKING:
    from core.discourse_intent import FollowUpClassification
    from core.discourse_state import DiscourseState

AnchorRejectReason = Literal[
    "empty",
    "pure_number",
    "measurement",
    "date",
    "percentage",
    "short_noun_phrase",
    "low_relevance",
    "deictic_phrase",
]

AnchorAcceptReason = Literal[
    "named_entity",
    "relevant_noun_phrase",
]

RewriteReason = Literal[
    "none",
    "query_substitution",
    "referent_anchor",
    "topic_anchor",
    "anchor_rejected",
    "not_follow_up",
    "conversation_health",
]

_PROPER_NAME = re.compile(
    r"\b((?:The\s+)?[A-Z][a-z0-9]+"
    r"(?:\s+(?:the|of|and|a|in)\s+[A-Za-z][a-z0-9]+|\s+[A-Z][a-z0-9]+)+)\b"
)
_TITLE_CASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_SINGLE_PROPER = re.compile(r"^[A-Z][a-zA-Z''\u2019-]{2,}$")

_PURE_NUMBER = re.compile(r"^[\d,]+(?:\.\d+)?$")
_PERCENT = re.compile(r"(?:^\d+(?:\.\d+)?\s*%|^\d+(?:\.\d+)?\s*percent\b)", re.I)
_MEASUREMENT = re.compile(
    r"(?:"
    r"^\d+(?:[.,]\d+)*\s*(?:%|percent|km²|km2|km|m²|m2|m|metres|meters|ft|feet|mi|miles|"
    r"kg|g|cm|mm|sq\.?\s*(?:km|mi|ft|m)?|square\s+(?:km|kilometres|kilometers|miles|feet))"
    r"|"
    r"\b(?:above|below)\s+sea\s+level\b"
    r"|"
    r"[≈~]\s*\d"
    r"|"
    r"\(\s*≈"
    r")",
    re.I,
)
_DATE = re.compile(
    r"(?:"
    r"\b\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?\b"
    r"|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|"
    r"\b\d{4}s\b"
    r")",
    re.I,
)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "its", "it",
    "this", "that", "what", "how", "when", "where", "who", "which", "about",
    "there", "here", "they", "them", "their", "our", "your", "commonly", "also",
})


@dataclass(frozen=True)
class AnchorScore:
    usable: bool
    confidence: float
    accept_reason: str = ""
    reject_reason: str = ""


@dataclass(frozen=True)
class DiscoursePromptRewrite:
    original: str
    grounded: str
    rewrite_anchor: Optional[str]
    rewrite_confidence: float
    rewrite_reason: str
    applied: bool

    def trace_fields(self) -> dict[str, object]:
        return {
            "rewrite_anchor": self.rewrite_anchor or "",
            "rewrite_confidence": round(self.rewrite_confidence, 3),
            "rewrite_reason": self.rewrite_reason,
        }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or "")


def _content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _tokenize(text) if t.lower() not in _STOPWORDS and len(t) >= 2}


def _reject_anchor_fragment(anchor: str) -> Optional[str]:
    """Return a reject reason for fragments that must never be anchored."""
    a = (anchor or "").strip()
    if not a:
        return "empty"
    if is_deictic_topic_phrase(a):
        return "deictic_phrase"
    if _PURE_NUMBER.match(a):
        return "pure_number"
    if _PERCENT.search(a):
        return "percentage"
    if _DATE.search(a):
        return "date"
    if _MEASUREMENT.search(a):
        return "measurement"
    return None


def validate_stored_discourse_topic(anchor: str) -> bool:
    """Whether a topic extracted from history may be kept in discourse state."""
    a = (anchor or "").strip()
    if _reject_anchor_fragment(a):
        return False
    if is_named_entity(a):
        return True
    return len(_tokenize(a)) >= 3


def is_named_entity(anchor: str) -> bool:
    a = (anchor or "").strip()
    if not a:
        return False
    if _SINGLE_PROPER.match(a):
        return True
    if _PROPER_NAME.search(a):
        return True
    if _TITLE_CASE.search(a):
        return True
    return False


def _semantic_relevance(anchor: str, user_message: str) -> bool:
    anchor_tokens = _content_tokens(anchor)
    user_tokens = _content_tokens(user_message)
    if anchor_tokens and user_tokens and (anchor_tokens & user_tokens):
        return True
    if is_named_entity(anchor) and (
        is_deictic_prompt(user_message) or has_possessive_anaphor(user_message)
    ):
        return True
    return False


def score_rewrite_anchor(anchor: str, *, user_message: str = "") -> AnchorScore:
    """
    Score whether ``anchor`` is safe to inject as a conversation anchor.

    Returns ``usable=True`` only for named entities or semantically relevant
    noun phrases (>= 3 tokens).
    """
    a = (anchor or "").strip()
    reject = _reject_anchor_fragment(a)
    if reject:
        return AnchorScore(False, 0.0, reject_reason=reject)

    if is_named_entity(a):
        conf = 0.90 if len(_tokenize(a)) >= 2 else 0.88
        return AnchorScore(True, conf, accept_reason="named_entity")

    tokens = _tokenize(a)
    if len(tokens) < 3:
        return AnchorScore(False, 0.0, reject_reason="short_noun_phrase")

    if _semantic_relevance(a, user_message):
        return AnchorScore(True, 0.76, accept_reason="relevant_noun_phrase")

    return AnchorScore(False, 0.0, reject_reason="low_relevance")


def _unchanged(original: str, reason: str) -> DiscoursePromptRewrite:
    return DiscoursePromptRewrite(
        original=original,
        grounded=original,
        rewrite_anchor=None,
        rewrite_confidence=0.0,
        rewrite_reason=reason,
        applied=False,
    )


def _anchor_from_substitutions(resolved: ResolvedUserQuery) -> Optional[str]:
    for _src, dst in resolved.substitutions:
        candidate = str(dst or "").strip()
        if candidate and score_rewrite_anchor(candidate, user_message=resolved.original).usable:
            return candidate
    return None


def resolve_discourse_prompt_rewrite(
    *,
    user_message: str,
    resolved_query: ResolvedUserQuery | None,
    follow_up: "FollowUpClassification",
    discourse: "DiscourseState | None",
    allow_rewrite: bool = True,
) -> DiscoursePromptRewrite:
    """
    Decide how to ground the latest user turn for prompt history.

    Original DB/UI text is unchanged; this only affects inference prompt assembly.
    """
    original = (user_message or "").strip()
    if not original:
        return _unchanged("", "none")

    if not allow_rewrite:
        return _unchanged(original, "conversation_health")

    if resolved_query is not None and resolved_query.succeeded:
        anchor = _anchor_from_substitutions(resolved_query)
        return DiscoursePromptRewrite(
            original=original,
            grounded=resolved_query.resolved.strip(),
            rewrite_anchor=anchor,
            rewrite_confidence=resolved_query.confidence,
            rewrite_reason=f"query_{resolved_query.rewrite_reason}",
            applied=True,
        )

    if not follow_up.active:
        return _unchanged(original, "not_follow_up")

    referent = (discourse.active_referent or "").strip() if discourse else ""
    if referent:
        score = score_rewrite_anchor(referent, user_message=original)
        if score.usable and score.confidence >= REWRITE_CONFIDENCE_MIN:
            return DiscoursePromptRewrite(
                original=original,
                grounded=f"[Referring to {referent}]\n\n{original}",
                rewrite_anchor=referent,
                rewrite_confidence=score.confidence,
                rewrite_reason="referent_anchor",
                applied=True,
            )

    topic = (discourse.active_topic or "").strip() if discourse else ""
    if topic and not is_deictic_topic_phrase(topic):
        score = score_rewrite_anchor(topic, user_message=original)
        if score.usable and score.confidence >= REWRITE_CONFIDENCE_MIN:
            return DiscoursePromptRewrite(
                original=original,
                grounded=f"[Continuing our discussion of {topic}]\n\n{original}",
                rewrite_anchor=topic,
                rewrite_confidence=score.confidence,
                rewrite_reason="topic_anchor",
                applied=True,
            )
        if score.reject_reason:
            return _unchanged(original, f"anchor_rejected:{score.reject_reason}")

    if referent or topic:
        return _unchanged(original, "anchor_rejected")

    return _unchanged(original, "none")


def select_salience_anchor(
    *,
    discourse: "DiscourseState | None",
    user_message: str,
    resolved_query: ResolvedUserQuery | None,
) -> tuple[Optional[str], float, str]:
    """Pick a validated anchor for system salience suffix, if any."""
    if resolved_query is not None and resolved_query.succeeded:
        return None, 0.0, "query_resolved"

    if discourse is None:
        return None, 0.0, "none"

    referent = (discourse.active_referent or "").strip()
    if referent:
        score = score_rewrite_anchor(referent, user_message=user_message)
        if score.usable and score.confidence >= REWRITE_CONFIDENCE_MIN:
            return referent, score.confidence, "referent_salience"

    topic = (discourse.active_topic or "").strip()
    if topic and not is_deictic_topic_phrase(topic):
        score = score_rewrite_anchor(topic, user_message=user_message)
        if score.usable and score.confidence >= REWRITE_CONFIDENCE_MIN:
            return topic, score.confidence, "topic_salience"
        if score.reject_reason:
            return None, 0.0, f"anchor_rejected:{score.reject_reason}"

    return None, 0.0, "none"
