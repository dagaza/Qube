"""
Lightweight per-session discourse state (active topic tracking).

Pure functions; no Qt, no LanceDB, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

TopicType = Literal["entity", "game", "concept", "task", "unknown"]

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


@dataclass(frozen=True)
class DiscourseState:
    active_topic: Optional[str] = None
    topic_type: TopicType = "unknown"
    last_explicit_turn_index: int = -1
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "discourse_topic": self.active_topic,
            "discourse_topic_type": self.topic_type,
            "discourse_topic_confidence": round(self.confidence, 3),
            "discourse_last_explicit_turn_index": self.last_explicit_turn_index,
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

    names = _PROPER_NAME.findall(s)
    if names:
        topic = max(names, key=len).strip()
        return topic[:120], _infer_topic_type(topic, s)

    titles = _TITLE_CASE.findall(s)
    if titles:
        topic = max(titles, key=len)
        return topic[:120], _infer_topic_type(topic, s)

    return None, "unknown"


def _assistant_topic_hint(content: str) -> tuple[Optional[str], TopicType]:
    s = (content or "").strip()
    if not s:
        return None, "unknown"
    first = re.split(r"[.!?\n]", s, maxsplit=1)[0]
    topic, ttype = _extract_topic_from_text(first)
    if topic:
        return topic, ttype
    names = _PROPER_NAME.findall(first)
    if names:
        topic = names[0].strip()
        return topic[:120], _infer_topic_type(topic, first)
    titles = _TITLE_CASE.findall(first)
    if titles:
        topic = titles[0]
        return topic[:120], _infer_topic_type(topic, first)
    return None, "unknown"


def update_discourse_state(
    history: list[dict[str, Any]],
    prior: DiscourseState | None,
    current_prompt: str,
) -> DiscourseState:
    """
    Derive active topic from explicit mentions in the current turn or prior context.
    """
    prior = prior or DiscourseState()
    turns = list(history or [])
    current_idx = len(turns) - 1 if turns else -1

    explicit, etype = _extract_topic_from_text(current_prompt)
    if explicit:
        return DiscourseState(
            active_topic=explicit,
            topic_type=etype,
            last_explicit_turn_index=current_idx,
            confidence=0.85,
        )

    if prior.active_topic and prior.last_explicit_turn_index >= 0:
        gap = current_idx - prior.last_explicit_turn_index
        if gap <= 6:
            decay = max(0.35, prior.confidence - 0.04 * max(0, gap - 1))
            return DiscourseState(
                active_topic=prior.active_topic,
                topic_type=prior.topic_type,
                last_explicit_turn_index=prior.last_explicit_turn_index,
                confidence=decay,
            )

    for i in range(len(turns) - 2, -1, -1):
        msg = turns[i]
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content") or "")
        if role == "user":
            topic, ttype = _extract_topic_from_text(content)
            if topic:
                if i + 1 < len(turns) and str(turns[i + 1].get("role", "")).lower() == "assistant":
                    asst = str(turns[i + 1].get("content") or "")
                    atype = _infer_topic_type(topic, asst)
                    if atype != "unknown":
                        ttype = atype
                return DiscourseState(
                    active_topic=topic,
                    topic_type=ttype,
                    last_explicit_turn_index=i,
                    confidence=0.70,
                )
        elif role == "assistant":
            topic, ttype = _assistant_topic_hint(content)
            if topic:
                return DiscourseState(
                    active_topic=topic,
                    topic_type=ttype,
                    last_explicit_turn_index=i,
                    confidence=0.55,
                )

    return DiscourseState()
