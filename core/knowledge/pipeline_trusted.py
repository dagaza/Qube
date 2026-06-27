"""Trusted knowledge pipeline: Wikipedia API + allowlisted DDG fallback."""

from __future__ import annotations

import time
from typing import Any

from core.knowledge.adapters.duckduckgo import (
    ADAPTER_ID as DDG_ADAPTER,
    is_failure_sentinel,
    search_duckduckgo,
)
from core.knowledge.adapters.wikipedia_api import ADAPTER_ID as WIKI_ADAPTER
from core.knowledge.adapters.wikipedia_api import search_wikipedia
from core.knowledge.bundle_builder import build_empty_bundle, build_trusted_knowledge_bundle
from core.knowledge.ranking.authority import is_allowlisted_url
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_TRUSTED_KNOWLEDGE
from core.retrieval_relevance import filter_web_results


class TrustedEvidencePipeline:
    """Wiki-first retrieval with gov/edu/wikipedia DDG fallback."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        query = ctx.query
        semantic = ctx.semantic_query or query
        budget = ctx.budget.max_results
        adapter_calls: list[str] = []
        raw_audit: list[dict[str, Any]] = []
        rejected_count = 0
        rel_diag: dict[str, Any] | None = None

        wiki_rows = search_wikipedia(query, max_results=min(2, budget))
        if wiki_rows:
            adapter_calls.append(WIKI_ADAPTER)
            raw_audit.extend(dict(r) for r in wiki_rows)

        kept: list[dict[str, Any]] = [dict(r) for r in wiki_rows]

        if len(kept) < budget:
            ddg_raw = search_duckduckgo(query, max_results=max(5, budget * 2))
            raw_audit.extend(dict(r) for r in ddg_raw)
            if not is_failure_sentinel(ddg_raw):
                if DDG_ADAPTER not in adapter_calls:
                    adapter_calls.append(DDG_ADAPTER)
                allowlisted = [
                    dict(r)
                    for r in ddg_raw
                    if is_allowlisted_url(str((r or {}).get("url") or ""))
                ]
                existing_urls = {
                    str(r.get("url") or "").strip().lower()
                    for r in kept
                    if r.get("url")
                }
                allowlisted = [
                    r
                    for r in allowlisted
                    if str(r.get("url") or "").strip().lower() not in existing_urls
                ]
                if allowlisted:
                    filtered, rel_diag = filter_web_results(
                        semantic,
                        allowlisted,
                        query_vector=ctx.query_vector,
                        embed_text_fn=ctx.embed_fn,
                        use_embedding_gate=True,
                    )
                    rejected_count = len(rel_diag.get("web_relevance_dropped") or [])
                    for row in filtered:
                        if len(kept) >= budget:
                            break
                        kept.append(dict(row))

        latency_ms = (time.time() - t0) * 1000

        if not kept:
            bundle = build_empty_bundle(
                query_raw=query,
                query_resolved=query,
                latency_ms=latency_ms,
                rejected_count=rejected_count,
                stop_reason="no_evidence",
                knowledge_service=SERVICE_TRUSTED_KNOWLEDGE,
            )
            return bundle, rel_diag, raw_audit

        stop_reason = (
            "sufficient_evidence"
            if len(kept) >= budget
            else "budget_exhausted"
        )
        bundle = build_trusted_knowledge_bundle(
            query_raw=query,
            query_resolved=query,
            kept_rows=kept,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=tuple(adapter_calls),
            stop_reason=stop_reason,
        )
        return bundle, rel_diag, raw_audit
