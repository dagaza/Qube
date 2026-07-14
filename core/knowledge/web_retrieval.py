"""Web retrieval entry points — legacy path and v2 evidence pipeline."""

from __future__ import annotations

import time
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from core.app_settings import entity_resolution_enabled, get_retrieval_profile
from core.knowledge.adapters.duckduckgo import is_failure_sentinel, search_duckduckgo
from core.knowledge.observability import build_retrieval_trace, record_retrieval_trace
from core.knowledge.registry import get_knowledge_service
from core.knowledge.retrieval_profiles import normalize_profile_id
from core.knowledge.retrieval_records import (
    build_context_fingerprint,
    save_retrieval_record,
)
from core.knowledge.types import (
    RetrievalBudget,
    RetrievalContext,
    SERVICE_GENERAL_WEB,
    WebRetrievalOutcome,
)
from core.retrieval_relevance import filter_web_results

if TYPE_CHECKING:
    from core.database import DatabaseManager


def run_legacy_web_retrieval(
    *,
    query: str,
    semantic_query: str,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    max_results: int = 3,
) -> WebRetrievalOutcome:
    """Original LLMWorker web path (search → sentinel → relevance gate)."""
    t0 = time.time()
    web_results = search_duckduckgo(query, max_results=max_results)
    web_results_raw_for_audit = [dict(r) for r in web_results]

    if is_failure_sentinel(web_results):
        return WebRetrievalOutcome(
            web_results=None,
            web_results_raw_for_audit=web_results_raw_for_audit,
            web_results_kept_for_audit=None,
            relevance_diag=None,
            skip_enrichment=True,
            bundle=None,
            latency_ms=(time.time() - t0) * 1000,
        )

    filtered, rel_diag = filter_web_results(
        semantic_query,
        web_results,
        query_vector=query_vector,
        embed_text_fn=embed_fn,
        use_embedding_gate=True,
    )

    if not filtered:
        return WebRetrievalOutcome(
            web_results=None,
            web_results_raw_for_audit=web_results_raw_for_audit,
            web_results_kept_for_audit=None,
            relevance_diag=rel_diag,
            skip_enrichment=True,
            bundle=None,
            latency_ms=(time.time() - t0) * 1000,
        )

    return WebRetrievalOutcome(
        web_results=filtered,
        web_results_raw_for_audit=web_results_raw_for_audit,
        web_results_kept_for_audit=[dict(r) for r in filtered],
        relevance_diag=rel_diag,
        skip_enrichment=False,
        bundle=None,
        latency_ms=(time.time() - t0) * 1000,
    )


def run_v2_web_retrieval(
    *,
    query: str,
    semantic_query: str,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    knowledge_service: str = SERVICE_GENERAL_WEB,
    adapter_filter: tuple[str, ...] | None = None,
    budget: RetrievalBudget | None = None,
    session_id: str | None = None,
    turn_id: int | None = None,
    library_store: Any | None = None,
    source_filter: str | None = None,
    preset_id: str | None = None,
    retrieval_profile: str | None = None,
    db: DatabaseManager | None = None,
) -> WebRetrievalOutcome:
    """Evidence pipeline path — legacy-compatible rows plus EvidenceBundle."""
    profile_id = normalize_profile_id(retrieval_profile or get_retrieval_profile())
    service = get_knowledge_service(knowledge_service)
    ctx = RetrievalContext(
        query=query,
        semantic_query=semantic_query,
        knowledge_service=service.id,
        query_vector=query_vector,
        embed_fn=embed_fn,
        budget=budget or service.default_budget(),
        adapter_filter=adapter_filter,
        library_store=library_store,
        source_filter=source_filter,
        preset_id=preset_id,
        retrieval_profile=profile_id,
    )
    bundle, rel_diag, raw_for_audit = service.retrieve(ctx)

    if rel_diag:
        http_summary = rel_diag.get("http_summary")
        if isinstance(http_summary, dict):
            from core.knowledge.provider_status import apply_http_summary

            apply_http_summary(http_summary)

    if entity_resolution_enabled() and bundle.sources:
        from core.knowledge.entities.enrich import enrich_bundle
        from core.knowledge.entities.pipeline import context_from_bundle

        bundle = enrich_bundle(bundle, context_from_bundle(bundle))

    kept_for_audit: list[dict[str, Any]] | None = None
    web_results: list[dict[str, Any]] | None = None

    if bundle.sources:
        web_results = [_evidence_to_legacy_row(s) for s in bundle.sources]
        kept_for_audit = [dict(r) for r in web_results]

    skip = not bundle.sources
    fingerprint = build_context_fingerprint(ctx, retrieval_profile=profile_id)
    pipeline_stages = None
    if rel_diag and isinstance(rel_diag.get("pipeline_stages"), list):
        pipeline_stages = rel_diag.get("pipeline_stages")

    trace = build_retrieval_trace(
        bundle,
        relevance_diag=rel_diag,
        session_id=session_id,
        turn_id=turn_id,
        preset_id=preset_id,
        retrieval_profile=profile_id,
        context_fingerprint=fingerprint.to_dict(),
        pipeline_stages=pipeline_stages,
    )
    record_retrieval_trace(trace, sources=bundle.sources)
    save_retrieval_record(
        db,
        request_id=trace.request_id,
        bundle=bundle,
        context_fingerprint=fingerprint,
        session_id=session_id,
        turn_id=turn_id,
    )

    return WebRetrievalOutcome(
        web_results=web_results,
        web_results_raw_for_audit=raw_for_audit,
        web_results_kept_for_audit=kept_for_audit,
        relevance_diag=rel_diag,
        skip_enrichment=skip,
        bundle=bundle,
        latency_ms=bundle.latency_ms,
    )


def run_web_retrieval(
    *,
    query: str,
    semantic_query: str,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    use_v2: bool = False,
    knowledge_service: str = SERVICE_GENERAL_WEB,
    adapter_filter: tuple[str, ...] | None = None,
    session_id: str | None = None,
    turn_id: int | None = None,
    library_store: Any | None = None,
    source_filter: str | None = None,
    preset_id: str | None = None,
    retrieval_profile: str | None = None,
    db: Any | None = None,
) -> WebRetrievalOutcome:
    if use_v2:
        return run_v2_web_retrieval(
            query=query,
            semantic_query=semantic_query,
            query_vector=query_vector,
            embed_fn=embed_fn,
            knowledge_service=knowledge_service,
            adapter_filter=adapter_filter,
            session_id=session_id,
            turn_id=turn_id,
            library_store=library_store,
            source_filter=source_filter,
            preset_id=preset_id,
            retrieval_profile=retrieval_profile,
            db=db,
        )
    return run_legacy_web_retrieval(
        query=query,
        semantic_query=semantic_query,
        query_vector=query_vector,
        embed_fn=embed_fn,
    )


def _evidence_to_legacy_row(source) -> dict[str, Any]:
    row: dict[str, Any] = {
        "title": source.title,
        "snippet": source.excerpt,
    }
    if source.url:
        row["url"] = source.url
    meta = source.raw_metadata or {}
    if meta.get("token_overlap") is not None:
        row["_web_token_overlap"] = meta["token_overlap"]
    if meta.get("semantic_score") is not None:
        row["_web_semantic_score"] = meta["semantic_score"]
    return row
