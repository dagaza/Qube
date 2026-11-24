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
from core.knowledge.ranking.web_gate import (
    TRUSTED_THRESHOLDS,
    drop_search_ads,
    dropped_for_diag,
    gate_rows,
)
from core.knowledge.search_outcome import attach_search_outcome
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_TRUSTED_KNOWLEDGE
from core.retrieval_query_cleanup import strip_request_frames
from core.retrieval_relevance import filter_web_results


# S12 (D19): DuckDuckGo search for @trusted is scoped to the allowlist's main
# zones (``site:wikipedia.org OR site:gov OR site:int OR site:edu <query>``).
# Live, on the 5 trusted fixtures: rows on the allowlist 16/30 unscoped vs
# 30/30 scoped, gate-kept rows 9 vs 12, sufficient 2 vs 4 of the 4 answerable
# fixtures; the fictional case stayed insufficient. A 6-site list (+ gov.uk,
# europa.eu) returned off-topic pages, so it is not used. Wikipedia fallback
# ignores the scope; the allowlist filter below still applies to every row.
TRUSTED_DISCOVERY_SITE_SCOPE: tuple[str, ...] = ("wikipedia.org", "gov", "int", "edu")


def _candidate_to_row(candidate: CandidateUrl) -> dict[str, Any]:
    row: dict[str, Any] = {
        "title": candidate.title or "",
        "snippet": candidate.snippet or "",
        "url": candidate.url,
    }
    # Plan S10 (R7): attribute the row to the provider that found it.
    if candidate.source:
        row["_discovery_provider"] = str(candidate.source)
    return row


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
        # Search and gate on the question, not its request frame (plan S5).
        search_query = strip_request_frames(query)
        semantic = strip_request_frames(ctx.semantic_query or query)
        budget = ctx.budget.max_results
        adapter_calls: list[str] = []
        raw_audit: list[dict[str, Any]] = []
        rejected_count = 0
        rel_diag: dict[str, Any] | None = None

        wiki_rows = search_wikipedia(search_query, max_results=min(2, budget))
        if wiki_rows:
            adapter_calls.append(WIKI_ADAPTER)
            raw_audit.extend(dict(r) for r in wiki_rows)

        # S7: Wikipedia rows pass the relevance gate like every other row (V13);
        # ``None`` means the reranker cannot run and the legacy path applies.
        wiki_copies = [dict(r) for r in wiki_rows]
        gate = gate_rows(semantic, wiki_copies, budget=budget, thresholds=TRUSTED_THRESHOLDS)
        kept: list[dict[str, Any]] = list(gate.kept) if gate is not None else [dict(r) for r in wiki_rows]

        if len(kept) < budget:
            discovery = discover_full_with_fallback(
                search_query,
                max_results=max(5, budget * 2),
                site_bias=TRUSTED_DISCOVERY_SITE_SCOPE,
                retrieval_profile=ctx.retrieval_profile,
            )
            ddg_raw = _discovery_rows(discovery)
            raw_audit.extend(ddg_raw)
            if not is_failure_sentinel(ddg_raw):
                provider = discovery.provider_id or DDG_ADAPTER
                if provider not in adapter_calls:
                    adapter_calls.append(provider)
                wiki_urls = {
                    str(r.get("url") or "").strip().lower() for r in wiki_rows if r.get("url")
                }
                # S8: the widened allowlist only where the reranker judges each row.
                allowlisted, _ads = drop_search_ads(
                    [
                        dict(r)
                        for r in ddg_raw
                        if is_allowlisted_url(str((r or {}).get("url") or ""), widened=gate is not None)
                        and str((r or {}).get("url") or "").strip().lower() not in wiki_urls
                    ]
                )
                if allowlisted and gate is not None:
                    merged = gate_rows(
                        semantic,
                        [*wiki_copies, *allowlisted],
                        budget=budget,
                        thresholds=TRUSTED_THRESHOLDS,
                    )
                    if merged is not None:
                        gate = merged
                        kept = list(gate.kept)
                    else:
                        gate = None
                        kept = [dict(r) for r in wiki_rows]
                if gate is None and any(
                    is_allowlisted_url(str(r.get("url") or ""), widened=False) for r in allowlisted
                ):
                    existing_urls = {
                        str(r.get("url") or "").strip().lower() for r in kept if r.get("url")
                    }
                    legacy_pool = [
                        r
                        for r in allowlisted
                        if str(r.get("url") or "").strip().lower() not in existing_urls
                        and is_allowlisted_url(str(r.get("url") or ""), widened=False)
                    ]
                    filtered, rel_diag = filter_web_results(
                        semantic,
                        legacy_pool,
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

        sufficiency, sufficiency_rationale, gate_warnings = "unscored", "", ()
        if gate is not None:
            rel_diag = dict(rel_diag or {})
            rel_diag["rerank"] = dict(gate.diag)
            rel_diag["sufficiency"] = gate.sufficiency.state.value
            if gate.dropped:
                rel_diag["rerank_dropped"] = dropped_for_diag(gate.dropped)
            rejected_count = len(gate.dropped)
            sufficiency = gate.sufficiency.state.value
            sufficiency_rationale = gate.sufficiency.rationale
            gate_warnings = tuple(gate.warnings)

        latency_ms = (time.time() - t0) * 1000

        if not kept:
            bundle = build_empty_bundle(
                query_raw=query,
                query_resolved=search_query,
                latency_ms=latency_ms,
                rejected_count=rejected_count,
                stop_reason="no_evidence",
                knowledge_service=SERVICE_TRUSTED_KNOWLEDGE,
                sufficiency=sufficiency,
                sufficiency_rationale=sufficiency_rationale,
            )
            return bundle, rel_diag, raw_audit

        stop_reason = (
            "sufficient_evidence"
            if len(kept) >= budget
            else "budget_exhausted"
        )
        bundle = build_trusted_knowledge_bundle(
            query_raw=query,
            query_resolved=search_query,
            kept_rows=kept,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=tuple(adapter_calls),
            stop_reason=stop_reason,
            sufficiency=sufficiency,
            sufficiency_rationale=sufficiency_rationale,
            extra_warnings=gate_warnings,
        )
        return bundle, rel_diag, raw_audit

        stop_reason = (
            "sufficient_evidence"
            if len(kept) >= budget
            else "budget_exhausted"
        )
        bundle = build_trusted_knowledge_bundle(
            query_raw=query,
            query_resolved=search_query,
            kept_rows=kept,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=tuple(adapter_calls),
            stop_reason=stop_reason,
        )
        return bundle, rel_diag, raw_audit
