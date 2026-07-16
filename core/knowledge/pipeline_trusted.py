"""Trusted knowledge pipeline: Wikipedia API + allowlisted DDG fallback."""

from __future__ import annotations

import time
from typing import Any

from core.knowledge.adapters.duckduckgo import ADAPTER_ID as DDG_ADAPTER, is_failure_sentinel
from core.knowledge.adapters.wikipedia_api import ADAPTER_ID as WIKI_ADAPTER
from core.knowledge.adapters.wikipedia_api import search_wikipedia
from core.knowledge.bundle_builder import build_empty_bundle, build_trusted_knowledge_bundle
from core.knowledge.discovery import discover_full_with_fallback
from core.knowledge.discovery.types import CandidateUrl
from core.knowledge.ranking.authority import is_allowlisted_url
from core.knowledge.search_outcome import attach_search_outcome
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_TRUSTED_KNOWLEDGE
from core.retrieval_relevance import filter_web_results


def _candidate_to_row(candidate: CandidateUrl) -> dict[str, Any]:
    return {
        "title": candidate.title or "",
        "snippet": candidate.snippet or "",
        "url": candidate.url,
    }


def _discovery_rows(discovery) -> list[dict[str, Any]]:
    rows = [_candidate_to_row(candidate) for candidate in discovery.candidates]
    if rows:
        return rows
    if discovery.raw_rows:
        return [dict(r) for r in discovery.raw_rows if isinstance(r, dict)]
    return []


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
            discovery = discover_full_with_fallback(
                query,
                max_results=max(5, budget * 2),
                retrieval_profile=ctx.retrieval_profile,
            )
            ddg_raw = _discovery_rows(discovery)
            raw_audit.extend(ddg_raw)
            if not is_failure_sentinel(ddg_raw):
                provider = discovery.provider_id or DDG_ADAPTER
                if provider not in adapter_calls:
                    adapter_calls.append(provider)
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

            if rel_diag is None:
                rel_diag = {}
            rel_diag["discovery_provider"] = discovery.provider_id
            if discovery.discovery_cache_hit:
                rel_diag["discovery_cache_hit"] = True
            if discovery.discovery_pace_wait_ms:
                rel_diag["discovery_pace_wait_ms"] = discovery.discovery_pace_wait_ms
            if discovery.privacy_tier:
                rel_diag["privacy_tier"] = discovery.privacy_tier
            rel_diag = attach_search_outcome(rel_diag, discovery.search_outcome)

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
