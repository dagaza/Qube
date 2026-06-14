"""Build read-only SkillContext from LLMWorker turn state."""

from __future__ import annotations

from typing import Any

from core.skills.types import SkillContext


def build_skill_context(
    *,
    user_query: str,
    clean_query: str,
    execution_route: str,
    all_ui_sources: list[dict[str, Any]] | None,
    follow_up_active: bool,
    explicit_remember_active: bool,
    file_search_active: bool,
    narrative_active: bool,
    decision: dict[str, Any] | None = None,
    query_embedding: Any | None = None,
    web_capability_blocked: bool = False,
    explicit_web_empty_results: bool = False,
    rag_capability_blocked: bool = False,
) -> SkillContext:
    sources = list(all_ui_sources or [])
    top_intent = None
    trace_summary = None
    if isinstance(decision, dict):
        ti = decision.get("top_intent")
        top_intent = str(ti) if ti is not None else None
        trace = decision.get("trace")
        if isinstance(trace, dict):
            reason = trace.get("winning_reason")
            if reason:
                trace_summary = str(reason)

    return SkillContext(
        user_query=str(user_query or ""),
        clean_query=str(clean_query or ""),
        execution_route=str(execution_route or "NONE").upper(),
        has_retrieval_sources=bool(sources),
        source_count=len(sources),
        follow_up_active=bool(follow_up_active),
        explicit_remember_active=bool(explicit_remember_active),
        file_search_active=bool(file_search_active),
        narrative_active=bool(narrative_active),
        web_capability_blocked=bool(web_capability_blocked),
        explicit_web_empty_results=bool(explicit_web_empty_results),
        rag_capability_blocked=bool(rag_capability_blocked),
        router_top_intent=top_intent,
        router_trace_summary=trace_summary,
        query_embedding=query_embedding,
    )
