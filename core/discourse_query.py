"""
Topic-aware query expansion for routing and retrieval (not user-visible text).
"""
from __future__ import annotations

from core.discourse_intent import FOLLOW_UP_SUPPRESS_THRESHOLD, FollowUpClassification
from core.discourse_state import DiscourseState


def resolve_routing_query(
    prompt: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
) -> str:
    """Semantic query for cognitive router embedding input."""
    return resolve_retrieval_query(prompt, follow_up, discourse)


def resolve_retrieval_query(
    prompt: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
) -> str:
    """Expanded query for memory/RAG/web search; original prompt if not a follow-up."""
    text = (prompt or "").strip()
    if not text:
        return text
    if follow_up.confidence < FOLLOW_UP_SUPPRESS_THRESHOLD:
        return text
    topic = (discourse.active_topic if discourse else None) or ""
    topic = topic.strip()
    if not topic:
        return text
    if topic.lower() in text.lower():
        return text
    return f"Regarding {topic}: {text}"


def resolve_web_query(
    prompt: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
) -> str:
    """Same topic expansion as retrieval — used before ``apply_tool_policy`` / search."""
    return resolve_retrieval_query(prompt, follow_up, discourse)


def should_veto_ungrounded_web_follow_up(
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
) -> bool:
    """
    Veto WEB only when a follow-up query is deictic and we cannot expand it
    with an active topic (search would be literally "tips for this").
    """
    if not follow_up.active:
        return False
    if discourse and (discourse.active_topic or "").strip():
        return False
    return follow_up.kind.value != "none"
