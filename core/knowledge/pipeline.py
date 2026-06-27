"""Evidence pipeline orchestrator (Phase 0: general web / DDG)."""

from __future__ import annotations

import time
from typing import Any

from core.knowledge.adapters.duckduckgo import (
    ADAPTER_ID,
    is_failure_sentinel,
    search_duckduckgo,
)
from core.knowledge.bundle_builder import build_empty_bundle, build_general_web_bundle
from core.knowledge.types import EvidenceBundle, RetrievalContext
from core.retrieval_relevance import filter_web_results


class EvidencePipeline:
    """Run adapter collection, relevance gate, and bundle assembly."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        query = ctx.query
        semantic = ctx.semantic_query or query

        raw_rows = search_duckduckgo(query, max_results=ctx.budget.max_results)
        latency_ms = (time.time() - t0) * 1000
        raw_copy = [dict(r) for r in raw_rows]

        if is_failure_sentinel(raw_rows):
            bundle = build_empty_bundle(
                query_raw=query,
                query_resolved=query,
                latency_ms=latency_ms,
                stop_reason="failure_sentinel",
            )
            return bundle, None, raw_copy

        filtered, rel_diag = filter_web_results(
            semantic,
            raw_rows,
            query_vector=ctx.query_vector,
            embed_text_fn=ctx.embed_fn,
            use_embedding_gate=True,
        )
        rejected_count = len(rel_diag.get("web_relevance_dropped") or [])

        if not filtered:
            bundle = build_empty_bundle(
                query_raw=query,
                query_resolved=query,
                latency_ms=latency_ms,
                rejected_count=rejected_count,
                stop_reason="relevance_filtered",
            )
            return bundle, rel_diag, raw_copy

        stop_reason = (
            "sufficient_evidence"
            if len(filtered) >= ctx.budget.max_results
            else "budget_exhausted"
        )
        bundle = build_general_web_bundle(
            query_raw=query,
            query_resolved=query,
            kept_rows=filtered,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=(ADAPTER_ID,),
            stop_reason=stop_reason,
        )
        return bundle, rel_diag, raw_copy
