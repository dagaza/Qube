"""Finance knowledge pipeline — configurable SEC and future finance adapters."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.app_settings import get_knowledge_source_preferences
from core.knowledge.adapters.registry import get_search_function
from core.knowledge.adapters.sec_edgar import ADAPTER_ID as SEC_ADAPTER
from core.knowledge.adapter_failures import adapter_error, failure_report, with_failure_note
from core.knowledge.http_metrics import begin_turn_http_metrics, snapshot_turn_http_summary
from core.knowledge.bundle_builder import build_empty_bundle, build_finance_knowledge_bundle
from core.knowledge.finance_query_planner import plan_finance_query
from core.knowledge.source_preferences import resolve_service_adapters
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_FINANCE_KNOWLEDGE
from core.knowledge.ranking.web_gate import FINANCE_THRESHOLDS, dropped_for_diag, gate_rows
from core.retrieval_relevance import filter_web_results


def _fetch_finance_rows(
    *,
    adapter_ids: tuple[str, ...],
    search_query: str,
    form_filter: tuple[str, ...],
    max_results: int,
    errors: list[dict[str, str]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    adapter_calls: list[str] = []
    rows: list[dict[str, Any]] = []
    per_adapter = max(2, max_results)

    with ThreadPoolExecutor(max_workers=min(3, len(adapter_ids) or 1)) as pool:
        futures = {}
        for aid in adapter_ids:
            search_fn = get_search_function(aid)
            if search_fn is None:
                continue
            if aid == SEC_ADAPTER:
                futures[
                    pool.submit(
                        search_fn,
                        search_query,
                        form_filter=form_filter,
                        max_results=per_adapter,
                    )
                ] = aid
            else:
                futures[
                    pool.submit(
                        search_fn,
                        search_query,
                        max_results=per_adapter,
                    )
                ] = aid
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


def _finish(bundle, rel_diag, http_summary, adapter_failures, failure_note):
    """S12: carry HTTP outcome and adapter failures into diag and rationale."""
    diag = dict(rel_diag or {})
    diag["http_summary"] = http_summary
    if adapter_failures:
        diag["adapter_failures"] = adapter_failures
    return with_failure_note(bundle, failure_note), diag


class FinanceEvidencePipeline:
    """Finance adapter retrieval with user-configurable sources."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        query = ctx.query
        semantic = ctx.semantic_query or query
        budget = ctx.budget.max_results
        plan = plan_finance_query(query, semantic_query=semantic)

        adapter_ids = resolve_service_adapters(
            SERVICE_FINANCE_KNOWLEDGE,
            query=query,
            composer_adapter_filter=ctx.adapter_filter,
            stored_preferences=get_knowledge_source_preferences(),
        )
        adapter_errors: list[dict[str, str]] = []
        begin_turn_http_metrics()
        adapter_calls, rows = _fetch_finance_rows(
            errors=adapter_errors,
            adapter_ids=adapter_ids,
            search_query=plan.search_query,
            form_filter=plan.form_types,
            max_results=max(3, budget),
        )
        http_summary = snapshot_turn_http_summary()
        adapter_failures, failure_note = failure_report(adapter_errors, http_summary)
        raw_audit: list[dict[str, Any]] = list(rows)
        rel_diag: dict[str, Any] | None = None
        rejected_count = 0

        kept: list[dict[str, Any]] = [dict(r) for r in rows]
        # S7: cross-encoder gate on the filings (``None`` → legacy path below).
        gate = gate_rows(
            plan.semantic_query or semantic,
            kept,
            budget=budget,
            thresholds=FINANCE_THRESHOLDS,
        ) if kept else None
        sufficiency, sufficiency_rationale, gate_warnings = "unscored", "", ()
        if gate is not None:
            rejected_count = len(gate.dropped)
            kept = [dict(r) for r in gate.kept]
            rel_diag = {
                "rerank": dict(gate.diag),
                "sufficiency": gate.sufficiency.state.value,
            }
            if gate.dropped:
                rel_diag["rerank_dropped"] = dropped_for_diag(gate.dropped)
            sufficiency = gate.sufficiency.state.value
            sufficiency_rationale = gate.sufficiency.rationale
            gate_warnings = tuple(gate.warnings)
        elif kept and len(kept) > budget:
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
                knowledge_service=SERVICE_FINANCE_KNOWLEDGE,
                sufficiency=sufficiency,
                sufficiency_rationale=sufficiency_rationale,
            )
            rel_diag = {
                **(rel_diag or {}),
                "finance_search_query": plan.search_query,
                "finance_form_types": list(plan.form_types),
                "finance_adapters_selected": list(adapter_ids),
            }
            bundle, rel_diag = _finish(bundle, rel_diag, http_summary, adapter_failures, failure_note)
            return bundle, rel_diag, raw_audit

        stop_reason = "sufficient_evidence" if len(kept) >= budget else "budget_exhausted"
        bundle = build_finance_knowledge_bundle(
            query_raw=query,
            query_resolved=plan.search_query,
            kept_rows=kept,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=tuple(adapter_calls),
            stop_reason=stop_reason,
            sufficiency=sufficiency,
            sufficiency_rationale=sufficiency_rationale,
            extra_warnings=gate_warnings,
        )
        rel_diag = {
            **(rel_diag or {}),
            "finance_search_query": plan.search_query,
            "finance_form_types": list(plan.form_types),
            "finance_adapters_selected": list(adapter_ids),
        }
        bundle, rel_diag = _finish(bundle, rel_diag, http_summary, adapter_failures, failure_note)
        return bundle, rel_diag, raw_audit
