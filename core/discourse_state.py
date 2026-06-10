"""
Lightweight per-session discourse state (active topic + referent tracking).

Pure functions; no Qt, no LanceDB, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional

from core.discourse_patterns import (
    has_possessive_anaphor,
    is_deictic_prompt,
    is_deictic_topic_phrase,
)
from core.discourse_prompt_rewrite import validate_stored_discourse_topic

TopicType = Literal["entity", "game", "concept", "task", "city", "person", "org", "unknown"]
ReferentSource = Literal[
    "assistant_answer",
    "assistant_pattern",
    "prior_session",
    "user_question",
    "history_scan",
    "none",
]

_GAME_HINTS = re.compile(
    r"\b(game|video game|roguelike|deckbuilder|playthrough|boss fight|card game)\b",
    re.I,
)
_ABOUT_CAPTURE = re.compile(
    r"\b(?:about|regarding|on)\s+(.{2,80}?)(?:\?|$|\.)",
    re.I,
)
_WHAT_IS_CAPTURE = re.compile(
    r"\b(?:what(?:'s|\s+is)|who(?:'s|\s+is)|tell me about|do you know about)\s+(.+?)\??\s*$",
    re.I,
)
_QUOTED = re.compile(r'"([^"]{2,120})"|\'([^\']{2,120})\'')
_TITLE_CASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
# Names with internal articles: Slay the Spire, Legend of Zelda, The Witcher
_PROPER_NAME = re.compile(
    r"\b((?:The\s+)?[A-Z][a-z0-9]+"
    r"(?:\s+(?:the|of|and|a|in)\s+[A-Za-z][a-z0-9]+|\s+[A-Z][a-z0-9]+)+)\b"
)
_TOPIC_CHANGE = re.compile(
    r"\b(?:let'?s talk about|switch(?:ing)? to|change topic to|new topic:?)\s+(.{2,80})",
    re.I,
)
_WHY_CAPTURE = re.compile(
    r"^\s*why\s+(?:do|does|is|are|can|could|would|will)\s+(.+?)\??\s*$",
    re.I,
)
_HOW_WORKS_CAPTURE = re.compile(
    r"^\s*how\s+(?:do|does)\s+(.+?)\s+work\??\s*$",
    re.I,
)
_HOW_CAPTURE = re.compile(
    r"^\s*how\s+(?:do|does|can|could|would|will)\s+(.+?)\??\s*$",
    re.I,
)
_SINGLE_PROPER_REFERENT = re.compile(
    r"^([A-Z][a-zA-Z''\u2019-]{1,50})[.!?]?$"
)
_CAPITAL_OF_HINT = re.compile(r"\bcapital\s+of\b", re.I)
_CITY_HINT = re.compile(r"\b(?:city|cities|town)\b", re.I)


@dataclass(frozen=True)
class DiscourseState:
    active_topic: Optional[str] = None
    topic_type: TopicType = "unknown"
    active_referent: Optional[str] = None
    referent_type: TopicType = "unknown"
    referent_source: ReferentSource = "none"
    referent_confidence: float = 0.0
    last_explicit_turn_index: int = -1
    confidence: float = 0.0

    def salience_anchor(self) -> Optional[str]:
        """Prefer resolved referent over topic for prompt injection."""
        ref = (self.active_referent or "").strip()
        if ref:
            return ref
        topic = (self.active_topic or "").strip()
        if topic and not is_deictic_topic_phrase(topic):
            return topic
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "discourse_topic": self.active_topic,
            "discourse_topic_type": self.topic_type,
            "discourse_topic_confidence": round(self.confidence, 3),
            "discourse_last_explicit_turn_index": self.last_explicit_turn_index,
            "discourse_referent": self.active_referent,
            "discourse_referent_type": self.referent_type,
            "discourse_referent_source": self.referent_source,
            "discourse_referent_confidence": round(self.referent_confidence, 3),
        }


def _infer_topic_type(topic: str, context: str = "") -> TopicType:
    blob = f"{topic} {context}".lower()
    if _GAME_HINTS.search(blob):
        return "game"
    if re.search(r"\b(project|task|todo|implement|build|fix)\b", blob):
        return "task"
    if re.search(r"\b(concept|idea|theory|principle|algorithm)\b", blob):
        return "concept"
    if topic and topic[0].isupper():
        return "entity"
    return "unknown"


def _infer_referent_type(
    referent: str,
    *,
    prior_user: str = "",
    current_prompt: str = "",
    assistant_context: str = "",
) -> TopicType:
    ctx = f"{prior_user} {current_prompt} {assistant_context}"
    if _CAPITAL_OF_HINT.search(ctx) or _CITY_HINT.search(ctx):
        return "city"
    return _infer_topic_type(referent, ctx)


def _normalize_concept_subject(subject: str) -> Optional[str]:
    s = (subject or "").strip(" .?!")
    if len(s) < 3:
        return None
    return s[:120]


def _extract_topic_from_text(text: str) -> tuple[Optional[str], TopicType]:
    s = (text or "").strip()
    if not s:
        return None, "unknown"

    m = _TOPIC_CHANGE.search(s)
    if m:
        topic = m.group(1).strip(" .?!")
        return topic[:120] or None, _infer_topic_type(topic, s)

    for q in _QUOTED.findall(s):
        topic = (q[0] or q[1] or "").strip()
        if topic:
            return topic[:120], _infer_topic_type(topic, s)

    m = _ABOUT_CAPTURE.search(s)
    if m:
        topic = m.group(1).strip(" .?!")
        if topic and len(topic) >= 2:
            return topic[:120], _infer_topic_type(topic, s)

    m = _WHAT_IS_CAPTURE.search(s)
    if m:
        topic = m.group(1).strip(" .?!")
        if topic and len(topic) >= 2:
            return topic[:120], _infer_topic_type(topic, s)

    m = _WHY_CAPTURE.search(s)
    if m:
        topic = _normalize_concept_subject(m.group(1))
        if topic:
            return topic, "concept"

    m = _HOW_WORKS_CAPTURE.search(s)
    if m:
        topic = _normalize_concept_subject(m.group(1))
        if topic:
            return topic, "concept"

    m = _HOW_CAPTURE.search(s)
    if m:
        topic = _normalize_concept_subject(m.group(1))
        if topic:
            return topic, "concept"

    names = _PROPER_NAME.findall(s)
    if names:
        topic = max(names, key=len).strip()
        return topic[:120], _infer_topic_type(topic, s)

    titles = _TITLE_CASE.findall(s)
    if titles:
        topic = max(titles, key=len)
        return topic[:120], _infer_topic_type(topic, s)

    return None, "unknown"


def extract_assistant_referent(content: str) -> Optional[str]:
    """Extract a primary entity referent from a short assistant answer."""
    s = (content or "").strip()
    if not s:
        return None
    first = re.split(r"[.!?\n]", s, maxsplit=1)[0].strip()
    if not first:
        return None

    m = _SINGLE_PROPER_REFERENT.match(first)
    if m:
        return m.group(1).strip()[:120]

    names = _PROPER_NAME.findall(first)
    if names:
        return max(names, key=len).strip()[:120]

    titles = _TITLE_CASE.findall(first)
    if titles:
        return max(titles, key=len).strip()[:120]

    return None


def _assistant_topic_hint(content: str) -> tuple[Optional[str], TopicType]:
    s = (content or "").strip()
    if not s:
        return None, "unknown"
    first = re.split(r"[.!?\n]", s, maxsplit=1)[0]
    topic, ttype = _extract_topic_from_text(first)
    if topic and validate_stored_discourse_topic(topic):
        return topic, ttype
    names = _PROPER_NAME.findall(first)
    if names:
        topic = names[0].strip()
        if validate_stored_discourse_topic(topic):
            return topic[:120], _infer_topic_type(topic, first)
    titles = _TITLE_CASE.findall(first)
    if titles:
        topic = titles[0]
        if validate_stored_discourse_topic(topic):
            return topic[:120], _infer_topic_type(topic, first)
    return None, "unknown"


def _last_assistant_before_current(turns: list[dict[str, Any]]) -> tuple[Optional[str], str]:
    """Return (assistant_content, preceding_user_content) before the last user turn."""
    if not turns:
        return None, ""
    end = len(turns)
    if str(turns[-1].get("role", "")).lower() == "user":
        end -= 1
    prior_user = ""
    for i in range(end - 1, -1, -1):
        role = str(turns[i].get("role", "")).lower()
        content = str(turns[i].get("content") or "").strip()
        if role == "assistant" and content:
            for j in range(i - 1, -1, -1):
                if str(turns[j].get("role", "")).lower() == "user":
                    prior_user = str(turns[j].get("content") or "").strip()
                    break
            return content, prior_user
    return None, ""


def _referent_from_assistant_content(
    asst_content: str,
    *,
    prior_user: str = "",
    current_prompt: str = "",
) -> tuple[Optional[str], TopicType, ReferentSource, float]:
    from core.discourse_answer_patterns import extract_referent_from_assistant_answer

    match = extract_referent_from_assistant_answer(
        asst_content, user_prompt=prior_user
    )
    if match:
        return match.referent, match.referent_type, "assistant_pattern", match.confidence

    referent = extract_assistant_referent(asst_content)
    if referent:
        return (
            referent,
            _infer_referent_type(
                referent,
                prior_user=prior_user,
                current_prompt=current_prompt,
                assistant_context=asst_content,
            ),
            "assistant_answer",
            0.80,
        )
    return None, "unknown", "none", 0.0


def _resolve_referent_from_history(
    history: list[dict[str, Any]],
    prior: DiscourseState,
    current_prompt: str,
    current_idx: int,
) -> tuple[Optional[str], TopicType, ReferentSource, float]:
    asst_content, prior_user = _last_assistant_before_current(history)
    if asst_content:
        referent, rtype, source, ref_conf = _referent_from_assistant_content(
            asst_content,
            prior_user=prior_user,
            current_prompt=current_prompt,
        )
        if referent:
            return referent, rtype, source, ref_conf

    if prior.active_referent and prior.last_explicit_turn_index >= 0:
        gap = current_idx - prior.last_explicit_turn_index
        if gap <= 6:
            return (
                prior.active_referent,
                prior.referent_type,
                prior.referent_source if prior.referent_source != "none" else "prior_session",
                prior.referent_confidence or prior.confidence,
            )

    if (
        prior.active_topic
        and not is_deictic_topic_phrase(prior.active_topic)
        and prior.last_explicit_turn_index >= 0
    ):
        gap = current_idx - prior.last_explicit_turn_index
        if gap <= 6:
            rtype: TopicType = prior.topic_type
            if rtype in ("entity", "game", "city", "person", "org"):
                return prior.active_topic, rtype, "prior_session", prior.confidence

    for i in range(len(history) - 2, -1, -1):
        msg = history[i]
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = str(msg.get("content") or "")
        topic, ttype = _extract_topic_from_text(content)
        if topic and not is_deictic_topic_phrase(topic):
            if i + 1 < len(history) and str(history[i + 1].get("role", "")).lower() == "assistant":
                asst = str(history[i + 1].get("content") or "")
                atype = _infer_topic_type(topic, asst)
                if atype != "unknown":
                    ttype = atype
            if ttype in ("entity", "game", "city", "person", "org"):
                return topic, ttype, "history_scan", 0.70

    return None, "unknown", "none", 0.0


def _state_with_topic(
    topic: str,
    ttype: TopicType,
    *,
    last_explicit_turn_index: int,
    confidence: float,
    referent_source: ReferentSource = "user_question",
    referent_confidence: float = 0.0,
) -> DiscourseState:
    active_referent: Optional[str] = None
    referent_type: TopicType = "unknown"
    ref_source: ReferentSource = "none"
    ref_conf = 0.0
    if ttype in ("entity", "game", "city", "person", "org"):
        active_referent = topic
        referent_type = ttype
        ref_source = referent_source
        ref_conf = referent_confidence or confidence
    return DiscourseState(
        active_topic=topic,
        topic_type=ttype,
        active_referent=active_referent,
        referent_type=referent_type,
        referent_source=ref_source,
        referent_confidence=ref_conf,
        last_explicit_turn_index=last_explicit_turn_index,
        confidence=confidence,
    )


def promote_referent_after_assistant(
    *,
    user_prompt: str,
    assistant_text: str,
    prior: DiscourseState | None,
) -> DiscourseState:
    """
    Promote a focal referent after an assistant answer (post-turn cache update).
    """
    prior = prior or DiscourseState()
    asst = (assistant_text or "").strip()
    if not asst:
        return prior

    referent, rtype, source, ref_conf = _referent_from_assistant_content(
        asst,
        prior_user=user_prompt,
        current_prompt=user_prompt,
    )
    if not referent:
        return prior

    preserved_topic = prior.active_topic
    if preserved_topic and is_deictic_topic_phrase(preserved_topic):
        preserved_topic = None

    topic_type = prior.topic_type
    if not preserved_topic and _CAPITAL_OF_HINT.search(user_prompt):
        topic_type = "city"

    return DiscourseState(
        active_topic=preserved_topic or referent,
        topic_type=topic_type if preserved_topic else rtype,
        active_referent=referent,
        referent_type=rtype,
        referent_source=source,
        referent_confidence=ref_conf,
        last_explicit_turn_index=prior.last_explicit_turn_index,
        confidence=max(prior.confidence, ref_conf),
    )


def update_discourse_state(
    history: list[dict[str, Any]],
    prior: DiscourseState | None,
    current_prompt: str,
) -> DiscourseState:
    """
    Derive active topic/referent from explicit mentions, prior context, or history.
    """
    prior = prior or DiscourseState()
    turns = list(history or [])
    current_idx = len(turns) - 1 if turns else -1
    current = (current_prompt or "").strip()

    explicit, etype = _extract_topic_from_text(current)
    if (
        explicit
        and not is_deictic_topic_phrase(explicit)
        and not has_possessive_anaphor(explicit)
    ):
        return _state_with_topic(
            explicit,
            etype,
            last_explicit_turn_index=current_idx,
            confidence=0.85,
            referent_source="user_question",
            referent_confidence=0.85,
        )

    deictic_turn = bool(
        explicit and (is_deictic_topic_phrase(explicit) or has_possessive_anaphor(explicit))
    ) or is_deictic_prompt(current)
    if deictic_turn:
        referent, rtype, ref_source, ref_conf = _resolve_referent_from_history(
            turns, prior, current, current_idx
        )
        if referent:
            preserved_topic = prior.active_topic
            if preserved_topic and is_deictic_topic_phrase(preserved_topic):
                preserved_topic = None
            active_topic = preserved_topic or referent
            topic_type = prior.topic_type if preserved_topic else rtype
            return DiscourseState(
                active_topic=active_topic,
                topic_type=topic_type,
                active_referent=referent,
                referent_type=rtype,
                referent_source=ref_source,
                referent_confidence=ref_conf,
                last_explicit_turn_index=prior.last_explicit_turn_index,
                confidence=max(0.80, ref_conf),
            )

    if prior.active_topic and prior.last_explicit_turn_index >= 0:
        gap = current_idx - prior.last_explicit_turn_index
        if gap <= 6:
            decay = max(0.35, prior.confidence - 0.04 * max(0, gap - 1))
            return replace(prior, confidence=decay)

    for i in range(len(turns) - 2, -1, -1):
        msg = turns[i]
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content") or "")
        if role == "user":
            topic, ttype = _extract_topic_from_text(content)
            if topic and not is_deictic_topic_phrase(topic):
                if i + 1 < len(turns) and str(turns[i + 1].get("role", "")).lower() == "assistant":
                    asst = str(turns[i + 1].get("content") or "")
                    atype = _infer_topic_type(topic, asst)
                    if atype != "unknown":
                        ttype = atype
                return _state_with_topic(
                    topic,
                    ttype,
                    last_explicit_turn_index=i,
                    confidence=0.70,
                )
        elif role == "assistant":
            topic, ttype = _assistant_topic_hint(content)
            if topic:
                return _state_with_topic(
                    topic,
                    ttype,
                    last_explicit_turn_index=i,
                    confidence=0.55,
                )

    return DiscourseState()
