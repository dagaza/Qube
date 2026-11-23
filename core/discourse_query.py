"""
Topic-aware query expansion for routing and retrieval (not user-visible text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.composer_attachments import is_web_composer_tool
from core.composer_skills import substantive_composer_prompt
from core.discourse_acronyms import expand_acronyms, extract_acronym_definitions
from core.discourse_intent import FOLLOW_UP_SUPPRESS_THRESHOLD, FollowUpClassification
from core.discourse_patterns import (
    has_possessive_anaphor,
    is_deictic_prompt,
    mentions_source_venue,
    strip_source_venue_phrases,
)
from core.discourse_prompt_rewrite import score_rewrite_anchor
from core.discourse_query_rewrite import ResolvedUserQuery, resolve_ambiguous_user_query
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

# "check that" / "is that true?" — matched on venue-stripped text and end-anchored,
# so "check that dark matter exists" keeps its own subject.
_DEICTIC_VERIFY_WEB = re.compile(
    r"(?:"
    r"\b(?:check|verify|confirm|validate|fact[\s-]?check|double[\s-]?check|"
    r"look\s+up|search(?:\s+for)?)\s+(?:this|that|it)"
    r"(?:\s+(?:up|out|again|too|for\s+me|please))*|"
    r"\b(?:is|was)\s+(?:this|that|it)\s+(?:really\s+)?"
    r"(?:true|correct|right|accurate|real)"
    r")\s*[?.!]*\s*$",
    re.I,
)
_WEB_TOOL_TOKEN_RE = re.compile(
    r"@\[tool:([^\]]+)\]|(?:(?<=\s)|^)@([a-z][\w-]*)",
    re.I,
)


@dataclass(frozen=True)
class SearchTargetResult:
    """Resolved search/retrieval string plus how it was derived (not user-visible)."""

    query: str
    rewrite_reason: str  # topic_expansion | referent_expansion | meta_prior_turn | none

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
    # (acronym, long form) pairs expanded into ``web_text`` from definitions
    # earlier in the conversation (plan S5). ``web_rewrite_reason`` is
    # ``"acronym_expansion"`` when that was the only rewrite; otherwise it keeps
    # the discourse reason and this field records the expansion.
    acronym_expansions: tuple[tuple[str, str], ...] = ()

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
        if self.acronym_expansions:
            out["web_query_acronym_expansions"] = [
                f"{acronym}={long_form}" for acronym, long_form in self.acronym_expansions
            ]
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
    web_tool_attached: bool = False,
) -> ResolvedRetrievalQuery:
    """
    Compose all channel-specific query strings from one inference base.

    ``inference_text`` is the post-``resolve_ambiguous_user_query`` string.
    Web search uses the same base plus ``history`` for meta-web prior-turn
    fallback (``meta_prior_turn``), detected on ``raw_text``, and for acronyms
    defined earlier in ``history`` (``acronym_expansion``, plan S5).
    """
    raw = (raw_text or "").strip()
    inference = (inference_text or raw).strip()
    web_target = resolve_search_target(
        inference,
        follow_up,
        discourse,
        history,
        meta_prompt=raw,
        web_tool_attached=web_tool_attached,
    )
    # An acronym defined earlier in the conversation is a safe signal on its
    # own, so this runs whether or not the turn was classified as a follow-up.
    web_text, expansions = expand_acronyms(
        web_target.query, extract_acronym_definitions(history)
    )
    web_reason = web_target.rewrite_reason
    if expansions and web_reason == "none":
        web_reason = "acronym_expansion"
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
        web_text=web_text,
        web_rewrite_reason=web_reason,
        inference_rewrite_reason=inf_reason,
        inference_confidence=inf_conf,
        acronym_expansions=tuple(expansions),
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


def _mentions_web_tool(text: str) -> bool:
    for m in _WEB_TOOL_TOKEN_RE.finditer(text):
        if is_web_composer_tool(m.group(1) or m.group(2) or ""):
            return True
    return False


def is_deictic_meta_web_request(prompt: str, *, web_tool_attached: bool = False) -> bool:
    """True when the user asks for web search but refers to a prior answer/topic deictically.

    ``web_tool_attached`` covers composer sends where ``@wikipedia`` etc. was lifted
    out of the prompt text into an attachment.
    """
    text = (prompt or "").strip()
    if not text:
        return False
    explicit = (
        web_tool_attached
        or detect_hard_explicit_web_request(text)
        or mentions_source_venue(text)
        or _mentions_web_tool(text)
    )
    if not explicit:
        return False
    if _DEICTIC_META_WEB.search(text):
        return True
    core = strip_source_venue_phrases(text)
    if not core:
        return False
    return bool(_DEICTIC_META_WEB.search(core) or _DEICTIC_VERIFY_WEB.search(core))


def prior_substantive_user_query(
    history: list[dict[str, Any]] | None,
    current_prompt: str,
) -> str | None:
    """Last prior user turn with real prompt text (skips meta-web and token-only sends)."""
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
        substantive = substantive_composer_prompt(content)
        if not substantive:
            continue
        return substantive
    return None


def _expand_query_with_discourse_anchor(
    query: str,
    discourse: DiscourseState | None,
    *,
    context_prompt: str = "",
) -> tuple[str, str]:
    """Attach active referent/topic when the query omits the entity name."""
    raw = (query or "").strip()
    if not raw or discourse is None:
        return raw, "none"
    anchor = ""
    rewrite_reason = "topic_expansion"
    referent = (fallback_referent(discourse) or "").strip()
    topic = (discourse.active_topic or "").strip()
    aspect = (discourse.active_aspect or "").strip()
    hint = (context_prompt or raw).strip()
    if referent:
        score = score_rewrite_anchor(referent, user_message=hint)
        if score.usable:
            anchor = referent
            rewrite_reason = "referent_expansion"
    elif topic and not is_deictic_topic_phrase(topic):
        score = score_rewrite_anchor(topic, user_message=hint)
        if score.usable:
            anchor = topic
    if anchor and anchor.lower() not in raw.lower():
        expansion = raw
        if aspect and aspect.lower() not in raw.lower():
            expansion = f"{aspect}: {raw}"
        return f"Regarding {anchor}: {expansion}", rewrite_reason
    return raw, "none"


def _discourse_at_prior_user_turn(
    history: list[dict[str, Any]] | None,
    prior_query: str,
) -> DiscourseState | None:
    """Rebuild discourse as it stood when the prior substantive user turn was asked."""
    if not history:
        return None
    prior = (prior_query or "").strip()
    if not prior:
        return None
    prior_substantive = substantive_composer_prompt(prior) or prior
    prior_index = -1
    for index, msg in enumerate(history):
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if substantive_composer_prompt(content) == prior_substantive or content == prior:
            prior_index = index
            break
    if prior_index <= 0:
        return None
    from core.discourse_state import update_discourse_state

    return update_discourse_state(history[:prior_index], None, prior)


def ground_prior_query_for_web_search(
    prior: str,
    discourse: DiscourseState | None,
    follow_up: FollowUpClassification,
) -> str:
    """Resolve deictic prior user text into a standalone web search query."""
    text = (prior or "").strip()
    if not text or discourse is None:
        return text
    if follow_up.active or is_deictic_prompt(text) or has_possessive_anaphor(text):
        rq = resolve_ambiguous_user_query(text, discourse, follow_up)
        if rq.succeeded:
            return rq.resolved.strip()
    if is_deictic_prompt(text) or has_possessive_anaphor(text):
        expanded, reason = _expand_query_with_discourse_anchor(text, discourse)
        if reason != "none":
            return expanded
    return text


def resolve_search_target(
    prompt: str,
    follow_up: FollowUpClassification,
    discourse: DiscourseState | None,
    history: list[dict[str, Any]] | None = None,
    *,
    meta_prompt: str | None = None,
    web_tool_attached: bool = False,
) -> SearchTargetResult:
    """Compose search/retrieval target from discourse expansion and meta-web fallbacks."""
    raw = (prompt or "").strip()
    if not raw:
        return SearchTargetResult("", "none")

    meta_text = (meta_prompt or "").strip() or raw
    if is_deictic_meta_web_request(meta_text, web_tool_attached=web_tool_attached):
        prior = prior_substantive_user_query(history, meta_text)
        if prior:
            search_discourse = discourse
            replayed = _discourse_at_prior_user_turn(history, prior)
            if replayed is not None and (
                (replayed.active_referent or "").strip()
                or (fallback_referent(replayed) or "").strip()
            ):
                search_discourse = replayed
            grounded = ground_prior_query_for_web_search(
                prior,
                search_discourse,
                follow_up,
            )
            return SearchTargetResult(grounded, "meta_prior_turn")

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
