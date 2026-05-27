"""
Assistive follow-up query expansion via the sidecar (never authoritative).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from core.app_settings import (
    get_sidecar_foreground_timeout_ms,
    get_sidecar_min_rewrite_confidence,
    get_sidecar_query_rewrite_enabled,
)
from core.discourse_intent import FollowUpClassification
from core.discourse_state import DiscourseState
from core.sidecar_types import QueryExpansion, SidecarTask

logger = logging.getLogger("Qube.SidecarQueryRewrite")

_PROPER_NAME = re.compile(
    r"\b((?:The\s+)?[A-Z][a-z0-9]+"
    r"(?:\s+(?:the|of|and|a|in)\s+[A-Za-z][a-z0-9]+|\s+[A-Z][a-z0-9]+)+)\b"
)


def _history_tail(history: list[dict[str, Any]], max_chars: int = 1200) -> str:
    lines: list[str] = []
    for msg in (history or [])[-4:]:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:400]}")
    text = "\n".join(lines)
    return text[-max_chars:]


def _anchor_blob(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def expansion_adds_unanchored_proper_nouns(expanded: str, *anchors: str) -> bool:
    """True when expansion introduces a proper-noun phrase absent from anchors."""
    blob = _anchor_blob(*anchors)
    for phrase in _PROPER_NAME.findall(expanded or ""):
        if phrase.strip() and phrase.lower() not in blob:
            return True
    return False


def propose_query_expansion(
    original: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
    history: list[dict[str, Any]] | None,
    sidecar_client: Any,
) -> Optional[QueryExpansion]:
    """
    Best-effort sidecar expansion. Returns None to keep regex-only retrieval.
    """
    text = (original or "").strip()
    if not text or not get_sidecar_query_rewrite_enabled():
        return None
    if not follow_up.active:
        return None
    if sidecar_client is None or not hasattr(sidecar_client, "complete"):
        return None
    if hasattr(sidecar_client, "available") and not sidecar_client.available:
        return None

    topic = (discourse.active_topic if discourse else None) or ""
    tail = _history_tail(history or [])

    timeout = get_sidecar_foreground_timeout_ms() / 1000.0
    try:
        result = sidecar_client.complete(
            SidecarTask.query_rewrite,
            timeout_sec=timeout,
            original_query=text,
            topic=topic,
            follow_up_kind=follow_up.kind.value,
            history_tail=tail,
        )
    except Exception as e:
        logger.debug("[Sidecar] query rewrite failed: %s", e)
        return None

    if not result.ok or not result.parsed:
        return None

    expanded = str(result.parsed.get("expanded_query") or "").strip()
    if not expanded or expanded.lower() == text.lower():
        return None

    conf = float(result.parsed.get("confidence") or result.confidence or 0.0)
    if conf < get_sidecar_min_rewrite_confidence():
        logger.info(
            "[Sidecar] rewrite below confidence floor (%.2f < min)",
            conf,
        )
        return None

    if expansion_adds_unanchored_proper_nouns(expanded, text, topic, tail):
        logger.info("[Sidecar] rewrite rejected (unanchored proper noun)")
        return None

    topic_source = str(result.parsed.get("topic_source") or "discourse_state")
    return QueryExpansion(
        original_query=text,
        expanded_query=expanded,
        confidence=conf,
        topic_source=topic_source,
    )
