"""General web evidence pipeline with optional selective page fetch."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from core.knowledge.adapters.duckduckgo import ADAPTER_ID, is_failure_sentinel
from core.knowledge.bundle_builder import (
    build_empty_bundle,
    build_fetched_general_web_bundle,
    build_general_web_bundle,
)
from core.knowledge.discovery import discover_full
from core.knowledge.discovery.types import CandidateUrl
from core.knowledge.egress_policy import DEFAULT_MAX_RESPONSE_BYTES_USER, EgressPolicy
from core.knowledge.extractors.registry import extract_document
from core.knowledge.fetch.engine import fetch_url
from core.knowledge.fetch.section_ranker import document_to_evidence_objects
from core.knowledge.fetch_provenance import (
    build_fetch_provenance,
    build_pipeline_stages_from_provenance,
)
from core.knowledge.search_outcome import (
    SearchOutcome,
    SearchOutcomeKind,
    attach_search_outcome,
)
from core.knowledge.ranking.recency import FRESH, STALE, assess_freshness
from core.knowledge.ranking.sufficiency import (
    Sufficiency,
    assess_sufficiency,
    relevance_probability,
)
from core.knowledge.ranking.web_gate import (
    WEB_THRESHOLDS,
    WARNING_RECENCY,
    WARNING_STALE,
    WARNING_UNDATED,
    WebGateResult,
    drop_search_ads,
    dropped_for_diag,
    final_sufficiency,
    gate_rows,
)
from core.knowledge.types import EvidenceBundle, EvidenceObject, RetrievalContext
from core.knowledge.web_fetch_context import resolve_web_fetch_options, resolve_web_relevance_options
from core.retrieval_query_cleanup import strip_request_frames
from core.retrieval_relevance import filter_web_results

# Gate mode that runs the S7 cross-encoder gate; "fetch_permissive" (@fetch with
# a URL) and "recipe_fetch_skip" keep their lenient legacy behaviour.
_CE_GATE_MODES = frozenset({"strict"})


def _candidate_to_row(candidate: CandidateUrl) -> dict[str, Any]:
    return {
        "title": candidate.title or "",
        "snippet": candidate.snippet or "",
        "url": candidate.url,
    }


def _is_ad_candidate(candidate: CandidateUrl) -> bool:
    from core.knowledge.ranking.web_gate import is_search_ad_url

    return is_search_ad_url(candidate.url)


def _rerank_skip_reason(question: str) -> str:
    from core.knowledge.ranking import rerank

    return rerank.unavailable_reason(question) or rerank.SKIP_ERROR


def _discovery_max_results(ctx: RetrievalContext, fetch_url_count: int) -> int:
    base = max(ctx.budget.max_results, fetch_url_count * 2, 3)
    return min(base, 8)


def _attach_fetch_provenance(
    rel_diag: dict[str, Any] | None,
    *,
    ctx: RetrievalContext,
    fetch_options,
    candidates: list[CandidateUrl],
    selected_urls: list[str],
    fetch_diag: dict[str, Any] | None,
    sections_emitted: int,
    fetch_url_count: int,
    rejected_count: int,
    latency_ms: float,
    discovery_provider: str = "duckduckgo",
) -> dict[str, Any]:
    diag = dict(rel_diag or {})
    provenance = build_fetch_provenance(
        query=ctx.query,
        composer_tool=fetch_options.composer_tool or ctx.composer_tool,
        site_bias=fetch_options.site_bias,
        discovery_provider=discovery_provider,
        candidates=candidates,
        selected_urls=selected_urls,
        fetch_diag=fetch_diag,
        sections_emitted=sections_emitted,
        fetch_url_count=fetch_url_count,
    )
    diag["fetch_provenance"] = provenance.to_dict()
    diag["fetch_url_count"] = fetch_url_count
    if fetch_options.site_bias:
        diag["site_bias"] = list(fetch_options.site_bias)
    if fetch_diag:
        diag["fetch"] = fetch_diag
    diag["pipeline_stages"] = build_pipeline_stages_from_provenance(
        provenance,
        rejected_count=rejected_count,
        latency_ms=latency_ms,
    )
    return diag


def _fetch_and_extract(
    ctx: RetrievalContext,
    *,
    candidates: list[CandidateUrl],
    fetch_url_count: int,
) -> tuple[list, dict[str, Any], list[str]]:
    from core.knowledge.types import EvidenceObject

    evidence: list[EvidenceObject] = []
    fetch_diag: dict[str, Any] = {
        "attempted": [],
        "succeeded": [],
        "failed": [],
        "attempts": [],
    }
    warnings: list[str] = []
    max_bytes = ctx.budget.max_fetch_bytes or DEFAULT_MAX_RESPONSE_BYTES_USER
    egress = EgressPolicy(max_response_bytes=max_bytes)

    for candidate in candidates[:fetch_url_count]:
        url = candidate.url
        fetch_diag["attempted"].append(url)
        result = fetch_url(
            url,
            egress_policy=egress,
            max_fetch_bytes=max_bytes,
            timeout=min(10.0, (ctx.budget.max_latency_ms or 8000) / 1000.0),
        )
        attempt_record = {
            "url": result.final_url or url,
            "tier": result.fetch_tier,
            "success": result.success,
            "failure_reason": result.failure_reason,
            "status_code": result.status_code,
            "total_bytes": result.total_bytes,
        }
        fetch_diag["attempts"].append(attempt_record)

        if not result.success or not result.html:
            fetch_diag["failed"].append(
                {
                    "url": url,
                    "failure_reason": result.failure_reason,
                    "status_code": result.status_code,
                    "tier": result.fetch_tier,
                    "total_bytes": result.total_bytes,
                }
            )
            continue

        try:
            document = extract_document(result.html, result.final_url or url)
        except RuntimeError as exc:
            message = str(exc).lower()
            failure_reason = (
                "extractor_unavailable"
                if "trafilatura is not installed" in message
                else "extract_failed"
            )
            fetch_diag["failed"].append(
                {
                    "url": url,
                    "failure_reason": failure_reason,
                    "status_code": result.status_code,
                    "tier": result.fetch_tier,
                    "total_bytes": result.total_bytes,
                }
            )
            attempt_record["failure_reason"] = failure_reason
            attempt_record["success"] = False
            continue

        sections = document_to_evidence_objects(
            document,
            query=ctx.query,
            semantic_query=ctx.semantic_query or ctx.query,
            query_vector=ctx.query_vector,
            embed_fn=ctx.embed_fn,
            max_results=ctx.budget.max_results,
        )
        if not sections:
            fetch_diag["failed"].append(
                {
                    "url": url,
                    "failure_reason": "empty_extract",
                    "tier": result.fetch_tier,
                    "total_bytes": result.total_bytes,
                }
            )
            attempt_record["failure_reason"] = "empty_extract"
            attempt_record["success"] = False
            continue

        metadata = document.metadata
        structured_type = None
        if document.structured_data:
            structured_type = str(document.structured_data.get("type") or "").strip() or None

        fetch_diag["succeeded"].append(
            {
                "url": result.final_url or url,
                "extractor": metadata.extractor_name if metadata else None,
                "extractor_version": metadata.extractor_version if metadata else None,
                "extractor_confidence": metadata.extractor_confidence if metadata else None,
                "section_count": len(sections),
                "document_sections": len(document.sections),
                "structured_data_type": structured_type,
                "total_bytes": result.total_bytes,
                "fetch_tier": result.fetch_tier,
            }
        )
        evidence.extend(sections)

    if fetch_diag["failed"] and fetch_diag["succeeded"]:
        warnings.append("partial_fetch")
    return evidence, fetch_diag, warnings


def _gate_fetched_sections(
    question: str,
    sections: list[EvidenceObject],
    *,
    serp_rows: list[dict[str, Any]],
    gate: WebGateResult,
) -> tuple[list[EvidenceObject], Any, tuple[str, ...], dict[str, Any]] | None:
    """Score fetched page sections with the S7 gate (``None`` if the model cannot run).

    Sections below ``keep`` are dropped. A section's freshness is its page's: the
    extractor's page date, the URL, or what the page's search snippet mentions.
    """
    from core.knowledge.ranking import rerank

    if not sections:
        return None
    texts = [
        f"{s.title}. {(s.full_text or s.excerpt or '').strip()}"[: rerank.DEFAULT_TEXT_CHARS]
        for s in sections
    ]
    result = rerank.rerank_texts(question, texts)
    if result is None:
        return None
    thresholds = WEB_THRESHOLDS
    need = gate.recency
    serp_by_url = {str(r.get("url") or ""): r for r in serp_rows}
    scored: list[tuple[float, EvidenceObject, dict[str, Any]]] = []
    for section, score in zip(sections, result.scores):
        row: dict[str, Any] = {"_rerank_score": round(float(score), 4)}
        if need is not None:
            serp = serp_by_url.get(str(section.url or ""), {})
            fresh = assess_freshness(
                {
                    "publication_date": section.publication_date,
                    "url": section.url,
                    "title": serp.get("title") or section.title,
                    "snippet": serp.get("snippet") or "",
                },
                window_days=need.window_days,
                today=None,
            )
            row["_freshness"] = fresh.status
            row["_source_date"] = fresh.source_date.iso() if fresh.source_date else None
            row["_source_date_basis"] = fresh.basis
        scored.append((float(score), section, row))
    survivors = [t for t in scored if t[0] >= thresholds.keep]
    survivors.sort(key=lambda t: -t[0])
    if need is not None:
        survivors.sort(key=lambda t: t[2].get("_freshness") == STALE)
    kept_rows = [t[2] for t in survivors]
    pool_result = assess_sufficiency([t[0] for t in scored], thresholds)
    final = final_sufficiency(kept_rows, pool_result, thresholds, need)
    kept: list[EvidenceObject] = []
    for score, section, row in survivors:
        meta = dict(section.raw_metadata or {})
        meta["rerank_score"] = row["_rerank_score"]
        if row.get("_freshness"):
            meta.update(
                freshness=row["_freshness"],
                source_date=row.get("_source_date"),
                source_date_basis=row.get("_source_date_basis"),
            )
        kept.append(
            replace(
                section,
                relevance_score=relevance_probability(score),
                reliability_score=max(0.0, min(1.0, relevance_probability(score) * 0.85)),
                raw_metadata=meta,
            )
        )
    warnings: tuple[str, ...] = ()
    if need is not None:
        statuses = [r.get("_freshness") for r in kept_rows]
        warnings = (WARNING_RECENCY,)
        if kept_rows and FRESH not in statuses:
            warnings += (WARNING_STALE if STALE in statuses else WARNING_UNDATED,)
    diag = {
        "sections_scored": len(scored),
        "sections_kept": len(kept),
        "latency_ms": round(result.latency_ms, 1),
        "section_scores": [round(t[0], 3) for t in scored],
        "state": final.state.value,
    }
    return kept, final, warnings, diag


def _discovery_rel_diag(
    search_outcome: SearchOutcome | None,
    *,
    raw_count: int,
    kept_count: int = 0,
    dropped: list | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "web_results_raw_count": raw_count,
        "web_results_kept_count": kept_count,
        "web_relevance_dropped": list(dropped or []),
    }
    if extra:
        diag.update(extra)
    return attach_search_outcome(diag, search_outcome)


def _discovery_metadata_extra(discovery) -> dict[str, Any]:
    extra: dict[str, Any] = {"discovery_provider": discovery.provider_id}
    if discovery.discovery_cache_hit:
        extra["discovery_cache_hit"] = True
    if discovery.discovery_pace_wait_ms:
        extra["discovery_pace_wait_ms"] = discovery.discovery_pace_wait_ms
    if getattr(discovery, "privacy_tier", None):
        extra["privacy_tier"] = discovery.privacy_tier
    return extra


def run_general_web_evidence_pipeline(
    ctx: RetrievalContext,
) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
    """Discover URLs, optionally fetch pages, and assemble an EvidenceBundle."""
    t0 = time.time()
    query = ctx.query
    # Search and gate on the question, not its request frame (plan S5).
    search_query = strip_request_frames(query)
    semantic = strip_request_frames(ctx.semantic_query or query)
    fetch_options = resolve_web_fetch_options(ctx)
    fetch_url_count = fetch_options.fetch_url_count
    relevance_options = resolve_web_relevance_options(ctx, fetch_options)

    discovery_limit = _discovery_max_results(ctx, fetch_url_count)
    discovery = discover_full(
        search_query,
        max_results=discovery_limit,
        site_bias=fetch_options.site_bias,
        retrieval_profile=ctx.retrieval_profile,
    )
    discovery_provider = discovery.provider_id
    discovery_extra = _discovery_metadata_extra(discovery)
    candidates = list(discovery.candidates)
    search_outcome = discovery.search_outcome
    raw_copy = [_candidate_to_row(c) for c in candidates]
    if not raw_copy and discovery.raw_rows:
        raw_copy = [dict(r) for r in discovery.raw_rows if isinstance(r, dict)]

    if not candidates:
        if is_failure_sentinel(raw_copy):
            latency_ms = (time.time() - t0) * 1000
            bundle = build_empty_bundle(
                query_raw=query,
                query_resolved=search_query,
                latency_ms=latency_ms,
                stop_reason="failure_sentinel",
            )
            rel_diag = _discovery_rel_diag(
                search_outcome,
                raw_count=len(raw_copy),
                extra=discovery_extra,
            )
            return bundle, rel_diag, raw_copy

    # Search-engine ads are never evidence (S7).
    raw_copy, ads_dropped = drop_search_ads(raw_copy)
    candidates = [c for c in candidates if not _is_ad_candidate(c)]
    gate: WebGateResult | None = None
    if relevance_options.apply_gate and relevance_options.mode in _CE_GATE_MODES:
        gate = gate_rows(
            semantic,
            [dict(r) for r in raw_copy],
            budget=len(raw_copy),
            thresholds=WEB_THRESHOLDS,
        )
    if gate is not None:
        filtered = list(gate.kept)
        rel_diag = {
            "web_results_raw_count": len(raw_copy),
            "web_results_kept_count": len(filtered),
            "web_relevance_min_overlap": None,
            "web_relevance_dropped": [
                {"title": str(r.get("title") or "")[:80], "rerank_score": r.get("_rerank_score")}
                for r in gate.dropped
            ],
            "web_relevance_embedding_gate": False,
            "web_relevance_gate_mode": "rerank",
            "rerank": dict(gate.diag),
            "sufficiency": gate.sufficiency.state.value,
        }
        if gate.dropped:
            rel_diag["rerank_dropped"] = dropped_for_diag(gate.dropped, adapter=ADAPTER_ID)
    elif relevance_options.apply_gate:
        filtered, rel_diag = filter_web_results(
            semantic,
            raw_copy,
            query_vector=ctx.query_vector,
            embed_text_fn=ctx.embed_fn,
            use_embedding_gate=relevance_options.use_embedding_gate,
            min_token_ratio=relevance_options.min_token_ratio,
        )
        rel_diag["web_relevance_gate_mode"] = relevance_options.mode
        if relevance_options.mode in _CE_GATE_MODES:
            rel_diag["rerank"] = {"skipped_reason": _rerank_skip_reason(semantic)}
    else:
        filtered = list(raw_copy)
        rel_diag = {
            "web_results_raw_count": len(raw_copy),
            "web_results_kept_count": len(filtered),
            "web_relevance_gate_mode": relevance_options.mode,
            "web_relevance_gate_skipped": True,
            "web_relevance_dropped": [],
            "web_relevance_embedding_gate": False,
        }
    if ads_dropped:
        rel_diag["web_ads_dropped"] = ads_dropped
    rel_diag = attach_search_outcome(rel_diag, search_outcome)
    rel_diag.update(discovery_extra)
    rejected_count = len(rel_diag.get("web_relevance_dropped") or [])
    latency_ms = (time.time() - t0) * 1000
    gate_sufficiency = gate.sufficiency.state.value if gate is not None else "unscored"
    gate_rationale = gate.sufficiency.rationale if gate is not None else ""
    gate_warnings = tuple(gate.warnings) if gate is not None else ()

    if not filtered:
        filtered_outcome = (
            SearchOutcome(
                kind=SearchOutcomeKind.RELEVANCE_FILTERED,
                provider=search_outcome.provider if search_outcome else "duckduckgo",
                http_status=search_outcome.http_status if search_outcome else None,
                parsed_rows=search_outcome.parsed_rows if search_outcome else 0,
                candidate_count=len(candidates),
                bot_challenge_signals=search_outcome.bot_challenge_signals
                if search_outcome
                else (),
                failure_sentinel_reason=search_outcome.failure_sentinel_reason
                if search_outcome
                else None,
                recovery_hint="All SERP rows were dropped by the relevance gate.",
            )
            if search_outcome
            else None
        )
        bundle = build_empty_bundle(
            query_raw=query,
            query_resolved=search_query,
            latency_ms=latency_ms,
            rejected_count=rejected_count,
            stop_reason="relevance_filtered",
            sufficiency=(
                gate_sufficiency
                if gate_sufficiency in (Sufficiency.INSUFFICIENT.value, Sufficiency.NONE.value)
                else "unscored"
            ),
            sufficiency_rationale=gate_rationale,
        )
        rel_diag = attach_search_outcome(rel_diag, filtered_outcome)
        return bundle, rel_diag, raw_copy

    if fetch_url_count <= 0:
        stop_reason = (
            "sufficient_evidence"
            if len(filtered) >= ctx.budget.max_results
            else "budget_exhausted"
        )
        bundle = build_general_web_bundle(
            query_raw=query,
            query_resolved=search_query,
            kept_rows=filtered,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            adapter_calls=(ADAPTER_ID,),
            stop_reason=stop_reason,
            sufficiency=gate_sufficiency,
            sufficiency_rationale=gate_rationale,
            extra_warnings=gate_warnings,
        )
        selected_urls = [
            str(row.get("url") or "").strip()
            for row in filtered
            if str(row.get("url") or "").strip()
        ]
        rel_diag = _attach_fetch_provenance(
            rel_diag,
            ctx=ctx,
            fetch_options=fetch_options,
            candidates=candidates,
            selected_urls=selected_urls,
            fetch_diag=None,
            sections_emitted=len(bundle.sources),
            fetch_url_count=0,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            discovery_provider=discovery_provider,
        )
        return bundle, rel_diag, raw_copy

    url_to_candidate = {c.url: c for c in candidates}
    ranked_candidates: list[CandidateUrl] = []
    for row in filtered:
        url = str(row.get("url") or "").strip()
        if url in url_to_candidate:
            ranked_candidates.append(url_to_candidate[url])
        elif url:
            ranked_candidates.append(
                CandidateUrl(
                    url=url,
                    title=str(row.get("title") or "") or None,
                    snippet=str(row.get("snippet") or "") or None,
                    source=ADAPTER_ID,
                )
            )

    evidence, fetch_diag, fetch_warnings = _fetch_and_extract(
        ctx,
        candidates=ranked_candidates,
        fetch_url_count=fetch_url_count,
    )
    selected_urls = [candidate.url for candidate in ranked_candidates[:fetch_url_count]]
    fetched_sufficiency, fetched_rationale, fetched_warnings = "unscored", "", ()
    if evidence and gate is not None:
        section_gate = _gate_fetched_sections(semantic, evidence, serp_rows=filtered, gate=gate)
        if section_gate is not None:
            kept_sections, section_suff, fetched_warnings, section_diag = section_gate
            rel_diag["rerank_sections"] = section_diag
            if kept_sections:
                evidence = kept_sections
                fetched_sufficiency = section_suff.state.value
                fetched_rationale = section_suff.rationale
            else:
                # The fetched page did not address the question: use the gated snippets.
                evidence = []
                fetch_warnings = [*fetch_warnings, "fetched_page_irrelevant"]
    latency_ms = (time.time() - t0) * 1000

    if evidence:
        bundle = build_fetched_general_web_bundle(
            query_raw=query,
            query_resolved=search_query,
            sources=evidence,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            fetch_attempts=len(fetch_diag.get("attempted") or []),
            fetch_successes=len(fetch_diag.get("succeeded") or []),
            warnings=tuple(dict.fromkeys((*fetch_warnings, *fetched_warnings))),
            sufficiency=fetched_sufficiency,
            sufficiency_rationale=fetched_rationale,
        )
        rel_diag = _attach_fetch_provenance(
            rel_diag,
            ctx=ctx,
            fetch_options=fetch_options,
            candidates=candidates,
            selected_urls=selected_urls,
            fetch_diag=fetch_diag,
            sections_emitted=len(evidence),
            fetch_url_count=fetch_url_count,
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            discovery_provider=discovery_provider,
        )
        return bundle, rel_diag, raw_copy

    warnings = list(fetch_warnings)
    warnings.append("snippet_fallback")
    stop_reason = (
        "sufficient_evidence"
        if len(filtered) >= ctx.budget.max_results
        else "budget_exhausted"
    )
    bundle = build_general_web_bundle(
        query_raw=query,
        query_resolved=search_query,
        kept_rows=filtered,
        rejected_count=rejected_count,
        latency_ms=latency_ms,
        adapter_calls=(ADAPTER_ID,),
        stop_reason=stop_reason,
        sufficiency=gate_sufficiency,
        sufficiency_rationale=gate_rationale,
        extra_warnings=gate_warnings,
    )
    if warnings:
        bundle = replace(
            bundle,
            warnings=tuple(dict.fromkeys((*bundle.warnings, *warnings))),
        )
    rel_diag = _attach_fetch_provenance(
        rel_diag,
        ctx=ctx,
        fetch_options=fetch_options,
        candidates=candidates,
        selected_urls=selected_urls,
        fetch_diag=fetch_diag,
        sections_emitted=len(bundle.sources),
        fetch_url_count=fetch_url_count,
        rejected_count=rejected_count,
        latency_ms=latency_ms,
        discovery_provider=discovery_provider,
    )
    return bundle, rel_diag, raw_copy
