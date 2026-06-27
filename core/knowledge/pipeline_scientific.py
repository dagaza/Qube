"""Scientific evidence pipeline: PubMed + OpenAlex + arXiv with Phase 3 ranking."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from core.knowledge.adapters.arxiv_api import ADAPTER_ID as ARXIV_ID
from core.knowledge.adapters.arxiv_api import search_arxiv
from core.knowledge.adapters.openalex import ADAPTER_ID as OPENALEX_ID
from core.knowledge.adapters.openalex import search_openalex
from core.knowledge.adapters.pubmed_eutils import ADAPTER_ID as PUBMED_ID
from core.knowledge.adapters.pubmed_eutils import search_pubmed
from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.bundle_builder import build_empty_bundle, build_scientific_evidence_bundle
from core.knowledge.conflicts.detect import detect_conflicts
from core.knowledge.evidence_cache import get_cached_rows, make_cache_key, set_cached_rows
from core.knowledge.ranking.diversity import mmr_select_rows
from core.knowledge.ranking.relevance import score_rows
from core.knowledge.ranking.reliability import apply_reliability_scores
from core.knowledge.ranking.stopping import adaptive_stop_reason
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_SCIENTIFIC_EVIDENCE

_DEFAULT_ADAPTERS = (PUBMED_ID, OPENALEX_ID, ARXIV_ID)

_ADAPTER_FNS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    PUBMED_ID: search_pubmed,
    OPENALEX_ID: search_openalex,
    ARXIV_ID: search_arxiv,
}

_MEDICAL_HINTS = re.compile(
    r"\b(drug|medication|medicine|disease|symptom|treatment|clinical|patient|"
    r"therapy|diagnosis|fda|vaccine|diabetes|cancer|ozempic|semaglutide)\b",
    re.IGNORECASE,
)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        doi = str(row.get("doi") or "").strip().lower()
        title_key = _normalize_title(str(row.get("title") or ""))
        if doi and doi in seen_doi:
            continue
        if title_key and title_key in seen_title:
            continue
        if doi:
            seen_doi.add(doi)
        if title_key:
            seen_title.add(title_key)
        kept.append(row)
    return kept


def _rank_candidates(
    rows: list[dict[str, Any]],
    *,
    ctx: RetrievalContext,
    max_results: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = sanitize_api_query(ctx.semantic_query or ctx.query)
    scored, rejected = score_rows(
        rows,
        query=query,
        query_vector=ctx.query_vector,
        embed_fn=ctx.embed_fn,
        min_score=0.12,
    )
    selected = mmr_select_rows(scored, max_results=max_results)
    selected = apply_reliability_scores(selected)
    return selected, rejected


class ScientificEvidencePipeline:
    """Parallel scientific adapter collection, Phase 3 rank, and bundle assembly."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        query = sanitize_api_query(ctx.query)
        semantic = sanitize_api_query(ctx.semantic_query or ctx.query)
        budget = ctx.budget.max_results
        adapter_ids = ctx.adapter_filter or _DEFAULT_ADAPTERS
        adapter_calls: list[str] = []
        raw_audit: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        cache_key = make_cache_key(
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            query=semantic or query,
            adapter_filter=adapter_ids,
        )
        cached = get_cached_rows(cache_key)
        if cached is not None:
            candidates = _dedupe_rows([dict(r) for r in cached])
            adapter_calls = sorted(
                {
                    str(r.get("_adapter"))
                    for r in candidates
                    if r.get("_adapter")
                }
            )
            raw_audit.extend(dict(r) for r in candidates)
        else:
            per_adapter = max(2, budget)
            with ThreadPoolExecutor(max_workers=min(3, len(adapter_ids))) as pool:
                futures = {
                    pool.submit(_ADAPTER_FNS[aid], query, max_results=per_adapter): aid
                    for aid in adapter_ids
                    if aid in _ADAPTER_FNS
                }
                for future in as_completed(futures):
                    aid = futures[future]
                    try:
                        rows = future.result()
                    except Exception:
                        rows = []
                    if rows:
                        adapter_calls.append(aid)
                        raw_audit.extend(dict(r) for r in rows)
                        candidates.extend(dict(r) for r in rows)
            adapter_calls = sorted(dict.fromkeys(adapter_calls))
            candidates = _dedupe_rows(candidates)
            if candidates:
                set_cached_rows(cache_key, candidates)

        ranked_ctx = RetrievalContext(
            query=query,
            semantic_query=semantic,
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            query_vector=ctx.query_vector,
            embed_fn=ctx.embed_fn,
            budget=ctx.budget,
            adapter_filter=ctx.adapter_filter,
        )
        kept, rejected = _rank_candidates(
            candidates, ctx=ranked_ctx, max_results=budget
        )
        latency_ms = (time.time() - t0) * 1000

        if not kept:
            return (
                build_empty_bundle(
                    query_raw=ctx.query,
                    query_resolved=semantic or query,
                    latency_ms=latency_ms,
                    rejected_count=len(rejected),
                    stop_reason="relevance_filtered",
                    knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
                ),
                {"scientific_relevance_dropped": len(rejected)},
                raw_audit,
            )

        avg_rel = sum(float(r.get("_scientific_relevance") or 0) for r in kept) / len(
            kept
        )
        abstract_count = sum(1 for r in kept if r.get("full_text"))
        adapter_count = len({str(r.get("_adapter") or "") for r in kept})
        stop_reason = adaptive_stop_reason(
            kept_count=len(kept),
            max_results=budget,
            avg_relevance=avg_rel,
            adapter_count=adapter_count,
            abstract_count=abstract_count,
        )

        bundle = build_scientific_evidence_bundle(
            query_raw=ctx.query,
            query_resolved=semantic or query,
            kept_rows=kept,
            rejected_count=len(rejected) + max(0, len(candidates) - len(kept)),
            latency_ms=latency_ms,
            adapter_calls=tuple(adapter_calls),
            stop_reason=stop_reason,
            medical_query=bool(_MEDICAL_HINTS.search(query)),
        )
        conflicts = detect_conflicts(bundle.sources, topic=query)
        if conflicts:
            bundle = _bundle_with_conflicts(bundle, conflicts)
        rel_diag = {
            "scientific_relevance_dropped": len(rejected),
            "scientific_avg_relevance": round(avg_rel, 4),
            "scientific_cache_hit": cached is not None,
        }
        return bundle, rel_diag, raw_audit


def _bundle_with_conflicts(bundle: EvidenceBundle, conflicts):
    warnings = tuple(dict.fromkeys((*bundle.warnings, "material_conflict")))
    return EvidenceBundle(
        bundle_id=bundle.bundle_id,
        query_raw=bundle.query_raw,
        query_resolved=bundle.query_resolved,
        knowledge_service=bundle.knowledge_service,
        retrieval_strategy=bundle.retrieval_strategy,
        profile_version=bundle.profile_version,
        retrieved_at=bundle.retrieved_at,
        latency_ms=bundle.latency_ms,
        confidence=bundle.confidence,
        coverage=bundle.coverage,
        coverage_rationale=bundle.coverage_rationale,
        authority_summary=bundle.authority_summary,
        reliability_summary=bundle.reliability_summary,
        diversity_summary=bundle.diversity_summary,
        sources=bundle.sources,
        rejected_count=bundle.rejected_count,
        warnings=warnings,
        conflicts=conflicts,
        stop_reason=bundle.stop_reason,
        adapter_calls=bundle.adapter_calls,
    )
