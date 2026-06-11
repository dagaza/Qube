"""
Referent stability policy: validation, entity/aspect parsing, replacement rules.

Pure functions; no Qt, no LanceDB, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.discourse_types import ReferentSource, TopicType

if TYPE_CHECKING:
    from core.discourse_state import DiscourseState

STICKY_USER_REFERENT_MIN_CONF = 0.80
ASSISTANT_PATTERN_REPLACE_MIN_CONF = 0.85

_POSSESSIVE_ENTITY_ASPECT = re.compile(
    r"\b(?:about|regarding|on)\s+([A-Za-z][A-Za-z0-9''\u2019-]*(?:\s+[A-Za-z][A-Za-z0-9''\u2019-]*)*)"
    r"(?:'s|'s|\u2019s)\s+(.{2,120}?)(?:\?|$|\.)",
    re.I,
)
_POSSESSIVE_HEAD = re.compile(
    r"^([A-Za-z][A-Za-z0-9''\u2019-]*(?:\s+[A-Za-z][A-Za-z0-9''\u2019-]*)*)"
    r"(?:'s|'s|\u2019s)\s+(.+?)\??\s*$",
    re.I,
)
_LIST_INTRODUCER = re.compile(
    r"\b(?:such as|like|including|featuring|e\.g\.|for example)\s+",
    re.I,
)
_PREP_TAIL = re.compile(
    r"\b(?:in|at|on|with|for|from|near|within|across|through)\s+[a-z]+\b",
    re.I,
)
_ENUM_AND_IN = re.compile(
    r"\b\w+\s+and\s+\w+\s+(?:in|at|on|with|for|from|near)\s+\w+",
    re.I,
)
_TOPIC_CHANGE = re.compile(
    r"\b(?:let'?s talk about|switch(?:ing)? to|change topic to|new topic:?)\s+",
    re.I,
)
_STOP_NOUNS = frozenset({
    "flowers", "flower", "trees", "tree", "species", "parks", "park",
    "examples", "items", "types", "kinds", "varieties", "plants", "animals",
    "fauna", "flora", "vegetation", "public", "private", "major", "minor",
})


@dataclass(frozen=True)
class EntityAspectParse:
    entity: Optional[str]
    aspect: Optional[str]
    topic_type: TopicType


def _normalize_entity(name: str) -> str:
    return (name or "").strip()[:120]


def _normalize_aspect(text: str) -> str:
    return (text or "").strip(" .?!")[:120]


def extract_entity_and_aspect(user_text: str) -> EntityAspectParse:
    """
    Parse possessive user turns into durable entity + facet.

    Example: "What about Kathmandu's flora and fauna?" → Kathmandu / flora and fauna
    """
    s = (user_text or "").strip()
    if not s:
        return EntityAspectParse(None, None, "unknown")

    m = _POSSESSIVE_ENTITY_ASPECT.search(s)
    if m:
        entity = _normalize_entity(m.group(1))
        aspect = _normalize_aspect(m.group(2))
        if entity and aspect:
            return EntityAspectParse(entity, aspect, _infer_entity_type(entity, s))

    m = _POSSESSIVE_HEAD.search(s)
    if m:
        entity = _normalize_entity(m.group(1))
        aspect = _normalize_aspect(m.group(2))
        if entity and aspect and len(aspect) >= 2:
            return EntityAspectParse(entity, aspect, _infer_entity_type(entity, s))

    return EntityAspectParse(None, None, "unknown")


def _infer_entity_type(entity: str, context: str = "") -> TopicType:
    from core.discourse_state import _infer_referent_type

    return _infer_referent_type(entity, prior_user=context, current_prompt=context)


def _appears_in_user_text(referent: str, user_texts: tuple[str, ...]) -> bool:
    r = (referent or "").strip().lower()
    if not r:
        return False
    blob = " ".join((t or "").lower() for t in user_texts)
    if r in blob:
        return True
    # Match entity head before possessive in user text
    head = r.split("'")[0].strip()
    return bool(head and head in blob)


def validate_referent_candidate(
    referent: str,
    *,
    user_message: str = "",
    user_prompt: str = "",
    assistant_text: str = "",
    source: ReferentSource = "none",
    user_history: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """
    Return (usable, reject_reason). Empty reject_reason when usable.
    """
    r = (referent or "").strip()
    if not r:
        return False, "empty"

    from core.discourse_prompt_rewrite import score_rewrite_anchor

    score = score_rewrite_anchor(r, user_message=user_message)
    if not score.usable:
        return False, score.reject_reason or "low_relevance"

    if _LIST_INTRODUCER.search(r):
        return False, "list_introducer"

    if _ENUM_AND_IN.search(r):
        return False, "enumeration_fragment"

    if _PREP_TAIL.search(r):
        return False, "preposition_tail"

    tokens = {t.lower() for t in re.findall(r"[A-Za-z']+", r)}
    if tokens & _STOP_NOUNS and source == "assistant_answer":
        return False, "stop_noun_fragment"

    user_texts = tuple(
        t for t in (user_message, user_prompt, *user_history) if (t or "").strip()
    )
    if source in ("assistant_answer", "history_scan") and user_texts:
        if not _appears_in_user_text(r, user_texts):
            tokens = re.findall(r"[A-Za-z']+", r)
            single_proper = (
                len(tokens) == 1
                and tokens[0][:1].isupper()
                and source == "assistant_answer"
            )
            if not single_proper:
                return False, "not_in_user_text"

    if source == "assistant_answer" and _LIST_INTRODUCER.search(assistant_text or ""):
        intro = _LIST_INTRODUCER.search(assistant_text or "")
        if intro and r.lower() in (assistant_text or "")[intro.end() :].lower():
            return False, "assistant_list_example"

    return True, ""


def should_replace_referent(
    prior: "DiscourseState | None",
    candidate: str,
    source: ReferentSource,
    confidence: float,
    *,
    user_prompt: str = "",
) -> tuple[bool, str]:
    """Return (allow, reason)."""
    prior = prior or None
    cand = (candidate or "").strip()
    if not cand:
        return False, "empty_candidate"

    if prior is None or not (prior.active_referent or "").strip():
        return True, "no_prior_referent"

    prior_ref = (prior.active_referent or "").strip()
    prior_source = prior.referent_source
    prior_conf = float(prior.referent_confidence or prior.confidence or 0.0)

    if cand.lower() == prior_ref.lower():
        return False, "same_referent"

    if _TOPIC_CHANGE.search(user_prompt or ""):
        return True, "explicit_topic_change"

    if source == "assistant_pattern" and confidence >= ASSISTANT_PATTERN_REPLACE_MIN_CONF:
        return True, "assistant_pattern"

    if (
        prior_source == "user_question"
        and prior_conf >= STICKY_USER_REFERENT_MIN_CONF
        and source in ("assistant_answer", "history_scan")
    ):
        return False, "sticky_user_referent"

    if (
        prior_source in ("user_question", "assistant_pattern", "prior_session")
        and prior_conf >= STICKY_USER_REFERENT_MIN_CONF
        and source == "assistant_answer"
    ):
        usable, reject = validate_referent_candidate(
            cand,
            user_prompt=user_prompt,
            source=source,
        )
        if not usable:
            return False, f"invalid_candidate:{reject}"
        if not _appears_in_user_text(cand, (user_prompt,)):
            return False, "assistant_not_in_user"

    return True, "allowed"


def fallback_referent(discourse: "DiscourseState | None") -> Optional[str]:
    """Last validated durable entity for rewrite/salience."""
    if discourse is None:
        return None
    ref = (discourse.active_referent or "").strip()
    if ref:
        usable, _ = validate_referent_candidate(
            ref,
            source=discourse.referent_source,
        )
        if usable:
            return ref
    return None


def rewrite_referent_target(discourse: "DiscourseState | None") -> Optional[str]:
    """
    Entity to substitute for possessive/deictic follow-ups.

    Always prefers durable ``active_referent`` over ``active_topic`` when topic
    holds aspect text.
    """
    if discourse is None:
        return None
    ref = fallback_referent(discourse)
    if ref:
        return ref
    topic = (discourse.active_topic or "").strip()
    if topic:
        usable, _ = validate_referent_candidate(topic, source=discourse.referent_source)
        if usable:
            return topic
    return None


def validate_resolved_query(
    resolved: str,
    discourse: "DiscourseState | None",
) -> tuple[bool, str]:
    """Post-substitution sanity check before treating rewrite as succeeded."""
    text = (resolved or "").strip()
    if not text:
        return False, "empty"

    target = rewrite_referent_target(discourse)
    if not target:
        return True, ""

    if target.lower() not in text.lower():
        return False, "substitution_target_missing"

    usable, reject = validate_referent_candidate(
        target,
        user_message=text,
        source=discourse.referent_source if discourse else "none",
    )
    if not usable:
        return False, reject or "invalid_target"

    if _ENUM_AND_IN.search(text) or _PREP_TAIL.search(target):
        return False, "fragment_in_query"

    return True, ""
