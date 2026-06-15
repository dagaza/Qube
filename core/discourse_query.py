"""
Topic-aware query expansion for routing and retrieval (not user-visible text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.discourse_intent import FOLLOW_UP_SUPPRESS_THRESHOLD, FollowUpClassification
from core.discourse_prompt_rewrite import score_rewrite_anchor
from core.discourse_query_rewrite import ResolvedUserQuery
from core.discourse_referent_policy import fallback_referent
from core.discourse_state import DiscourseState, is_deictic_topic_phrase
from core.memory_filters import detect_hard_explicit_web_request

# Deictic follow-ups that ask for web search without restating the topic.
_DEICTIC_META_WEB = re.compile(
    r"(?:"
    r"\bfor\s+the\s+answer\b|"
    r"\bsearch\s+for\s+the\s+answer\b|"
    r"\bonline\s+search\s+for\s+the\s+answer\b|"
    r"\b(?:do\s+)?an?\s+online\s+search\s+for\s+the\s+answer\b|"
    r"\bfor\s+(?:this|that|it)\b(?!\s+\w)|"
    r"\babout\s+(?:this|that|it)\b(?!\s+\w)|"
    r"\blook\s+(?:it|that|this)\s+up\b"
    r")",
    re.I,
)


@dataclass(frozen=True)
class SearchTargetResult:
    """Resolved search/retrieval string plus how it was derived (not user-visible)."""

    query: str
    rewrite_reason: str  # topic_expansion | meta_prior_turn | none

    @property
    def rewritten(self) -> bool:
        return self.rewrite_reason != "none"


@dataclass(frozen=True)
class ResolvedRetrievalQuery:
    """
    Canonical per-turn query representation for routing and retrieval.

    Built once after discourse classification; all retrieval subsystems
    should consume these fields instead of re-deriving from ``raw_text``.
    """

    raw_text: str
    inference_text: str
    routing_text: str
    retrieval_text: str
    web_text: str
    web_rewrite_reason: str = "none"
    inference_rewrite_reason: str = "none"
    inference_confidence: float = 0.0

    @property
    def web_rewritten(self) -> bool:
        return self.web_rewrite_reason != "none"

    @property
    def inference_rewritten(self) -> bool:
        return self.inference_rewrite_reason != "none"

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Flat decision/routing-debug fields (additive, JSON-safe)."""
        out: dict[str, Any] = {
            "resolved_query_raw": self.raw_text,
            "resolved_query_inference": self.inference_text,
            "resolved_query_routing": self.routing_text,
            "resolved_query_retrieval": self.retrieval_text,
            "resolved_query_web": self.web_text,
            "web_query_rewrite_reason": self.web_rewrite_reason,
        }
        if self.inference_rewritten:
            out["inference_rewrite_reason"] = self.inference_rewrite_reason
            out["inference_rewrite_confidence"] = round(self.inference_confidence, 3)
        return out


def build_resolved_retrieval_query(
    *,
    raw_text: str,
    inference_text: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
    history: list[dict[str, Any]] | None = None,
    resolved_query: ResolvedUserQuery | None = None,
) -> ResolvedRetrievalQuery:
    """
    Compose all channel-specific query strings from one inference base.

    ``inference_text`` is the post-``resolve_ambiguous_user_query`` string.
    Web search uses the same base plus ``history`` for meta-web prior-turn
    fallback (``meta_prior_turn``).
    """
    raw = (raw_text or "").strip()
    inference = (inference_text or raw).strip()
    web_target = resolve_search_target(inference, follow_up, discourse, history)
    inf_reason = "none"
    inf_conf = 0.0
    if resolved_query is not None and resolved_query.succeeded:
        inf_reason = resolved_query.rewrite_reason
        inf_conf = float(resolved_query.confidence or 0.0)
    return ResolvedRetrievalQuery(
        raw_text=raw,
        inference_text=inference,
        routing_text=resolve_routing_query(inference, follow_up, discourse),
        retrieval_text=resolve_retrieval_query(inference, follow_up, discourse),
        web_text=web_target.query,
        web_rewrite_reason=web_target.rewrite_reason,
        inference_rewrite_reason=inf_reason,
        inference_confidence=inf_conf,
    )


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
    return resolve_search_target(prompt, follow_up, discourse, None).query


def is_deictic_meta_web_request(prompt: str) -> bool:
    """True when the user asks for web search but refers to a prior answer/topic deictically."""
    text = (prompt or "").strip()
    if not text or not detect_hard_explicit_web_request(text):
        return False
    return bool(_DEICTIC_META_WEB.search(text))


def prior_substantive_user_query(
    history: list[dict[str, Any]] | None,
    current_prompt: str,
) -> str | None:
    """Last prior user turn that is not the current prompt or another meta web request."""
    if not history:
        return None
    current = (current_prompt or "").strip()
    for msg in reversed(history):
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if not content or content == current:
            continue
        if is_deictic_meta_web_request(content):
            continue
        return content
    return None


def resolve_search_target(
    prompt: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
    history: list[dict[str, Any]] | None = None,
) -> SearchTargetResult:
    """Compose search/retrieval target from discourse expansion and meta-web fallbacks."""
    raw = (prompt or "").strip()
    if not raw:
        return SearchTargetResult("", "none")

    if is_deictic_meta_web_request(prompt):
        prior = prior_substantive_user_query(history, prompt)
        if prior:
            return SearchTargetResult(prior, "meta_prior_turn")

    if follow_up.confidence >= FOLLOW_UP_SUPPRESS_THRESHOLD:
        anchor = ""
        rewrite_reason = "topic_expansion"
        if discourse:
            referent = (fallback_referent(discourse) or "").strip()
            topic = (discourse.active_topic or "").strip()
            aspect = (discourse.active_aspect or "").strip()
            if referent:
                score = score_rewrite_anchor(referent, user_message=raw)
                if score.usable:
                    anchor = referent
                    rewrite_reason = "referent_expansion"
            elif topic and not is_deictic_topic_phrase(topic):
                score = score_rewrite_anchor(topic, user_message=raw)
                if score.usable:
                    anchor = topic
        if anchor and anchor.lower() not in raw.lower():
            expansion = raw
            if aspect and aspect.lower() not in raw.lower():
                expansion = f"{aspect}: {raw}"
            return SearchTargetResult(
                f"Regarding {anchor}: {expansion}",
                rewrite_reason,
            )

    return SearchTargetResult(raw, "none")


def web_query_rewrite_failed(
    prompt: str,
    follow_up: FollowUpClassification,
    resolved_query: str,
    *,
    explicit_web: bool,
) -> bool:
    """True when an explicit web follow-up still searches with the raw meta phrase."""
    raw = (prompt or "").strip()
    resolved = (resolved_query or "").strip()
    if not explicit_web or not raw or resolved != raw:
        return False
    return bool(is_deictic_meta_web_request(prompt) or follow_up.active)


def resolve_web_query(
    prompt: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
    history: list[dict[str, Any]] | None = None,
    *,
    inference_text: str | None = None,
) -> str:
    """Topic expansion + meta web-request rewrite before ``apply_tool_policy`` / search."""
    base = (inference_text or prompt or "").strip() or (prompt or "").strip()
    return resolve_search_target(base, follow_up, discourse, history).query


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
    if discourse and (
        (discourse.active_referent or "").strip()
        or (discourse.active_topic or "").strip()
    ):
        return False
    return follow_up.kind.value != "none"
