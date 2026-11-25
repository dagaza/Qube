"""Legal knowledge pipeline — configurable CourtListener and future legal adapters."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.app_settings import get_knowledge_source_preferences
from core.knowledge.adapters.registry import get_search_function
from core.knowledge.adapter_failures import adapter_error, failure_report, with_failure_note
from core.knowledge.adapter_key_gate import split_keyless
from core.knowledge.http_metrics import begin_turn_http_metrics, snapshot_turn_http_summary
from core.knowledge.bundle_builder import build_empty_bundle, build_legal_knowledge_bundle
from core.knowledge.legal_query_planner import plan_legal_query
from core.knowledge.source_preferences import resolve_service_adapters
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_LEGAL_KNOWLEDGE
from core.retrieval_relevance import filter_web_results


def _fetch_legal_rows(
    *,
    adapter_ids: tuple[str, ...],
    search_query: str,
    max_results: int,
    errors: list[dict[str, str]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    adapter_calls: list[str] = []
    rows: list[dict[str, Any]] = []
    per_adapter = max(2, max_results)

    with ThreadPoolExecutor(max_workers=min(3, len(adapter_ids) or 1)) as pool:
        futures = {
            pool.submit(
                get_search_function(aid),
                search_query,
                max_results=per_adapter,
            ): aid
            for aid in adapter_ids
            if get_search_function(aid) is not None
        }
        for future in as_completed(futures):
            aid = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # S12: recorded, not swallowed (§3.7)
                result = []
                if errors is not None:
                    errors.append(adapter_error(aid, exc))
            if result:
                adapter_calls.append(aid)
                rows.extend(dict(r) for r in result)
    return sorted(dict.fromkeys(adapter_calls)), rows


def _finish(bundle, rel_diag, http_summary, adapter_failures, failure_note, adapter_skips=()):
    """S12: carry HTTP outcome and adapter failures into diag and rationale.

    S22 (D35): adapters left out for a missing API key are named in
    ``adapter_skips``.
    """
    diag = dict(rel_diag or {})
    diag["http_summary"] = http_summary
    if adapter_failures:
        diag["adapter_failures"] = adapter_failures
    if adapter_skips:
        diag["adapter_skips"] = list(adapter_skips)
    return with_failure_note(bundle, failure_note), diag


class LegalEvidencePipeline:
    """Legal adapter retrieval with user-configurable sources."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        query = ctx.query
        semantic = ctx.semantic_query or query
        budget = ctx.budget.max_results
        plan = plan_legal_query(query, semantic_query=semantic)

        adapter_ids = resolve_service_adapters(
            SERVICE_LEGAL_KNOWLEDGE,
            query=query,
            composer_adapter_filter=ctx.adapter_filter,
            stored_preferences=get_knowledge_source_preferences(),
        )
        adapter_ids, adapter_skips = split_keyless(adapter_ids)
        adapter_errors: list[dict[str, str]] = []
        begin_turn_http_metrics()
        adapter_calls, rows = _fetch_legal_rows(
            errors=adapter_errors,
            adapter_ids=adapter_ids,
            search_query=plan.search_query,
            max_results=max(3, budget),
        )
        http_summary = snapshot_turn_http_summary()
        adapter_failures, failure_note = failure_report(adapter_errors, http_summary)
        raw_audit: list[dict[str, Any]] = list(rows)
        rel_diag: dict[str, Any] | None = None
        rejected_count = 0

        kept: list[dict[str, Any]] = [dict(r) for r in rows]
        if kept and len(kept) > budget:
            filtered, rel_diag = filter_web_results(
                plan.semantic_query,
                kept,
                query_vector=ctx.query_vector,
                embed_text_fn=ctx.embed_fn,
                use_embedding_gate=False,
            )
            rejected_count = max(0, len(kept) - len(filtered))
            kept = [dict(r) for r in filtered[:budget]]

        latency_ms = (time.time() - t0) * 1000
        if not kept:
            bundle = build_empty_bundle(
                query_raw=query,
                query_resolved=plan.search_query,
                latency_ms=latency_ms,
                rejected_count=rejected_count,
                stop_reason="no_evidence",
                knowledge_service=SERVICE_LEGAL_KNOWLEDGE,
            )
            rel_diag = {
                **(rel_diag or {}),
                "legal_search_query": plan.search_query,
                "legal_adapters_selected": list(adapter_ids),
            }
            bundle, rel_diag = _finish(
                bundle, rel_diag, http_summary, adapter_failures, failure_note, adapter_skips
            )
            return bundle, rel_diag, raw_audit

        stop_reason = "sufficient_evidence" if len(kept) >= budget else "budget_exhausted"
        bundle = build_legal_knowledge_bundle(
            query_raw=query,
            query_resolved=plan.search_query,
            kept_rows=kept,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=tuple(adapter_calls),
            stop_reason=stop_reason,
        )
        rel_diag = {
            **(rel_diag or {}),
            "legal_search_query": plan.search_query,
            "legal_adapters_selected": list(adapter_ids),
        }
        bundle, rel_diag = _finish(
            bundle, rel_diag, http_summary, adapter_failures, failure_note, adapter_skips
        )
        return bundle, rel_diag, raw_audit
