"""Scientific evidence pipeline: PubMed + OpenAlex + arXiv with Phase 3 ranking."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.adapters.registry import get_search_function
from core.app_settings import get_knowledge_source_preferences
from core.knowledge.bundle_builder import build_empty_bundle, build_scientific_evidence_bundle
from core.knowledge.conflicts.detect import detect_conflicts
from core.knowledge.scientific_adapters import is_medical_query
from core.knowledge.scientific_discipline import detect_scientific_discipline
from core.knowledge.scientific_query_planner import adapter_query_for, plan_scientific_query
from core.knowledge.source_preferences import resolve_service_adapters
from core.knowledge.evidence_cache import get_cached_rows, make_cache_key, set_cached_rows
from core.knowledge.ranking.diversity import mmr_select_rows
from core.knowledge.ranking.relevance import score_rows
from core.knowledge.ranking.trial_grounding import extract_trial_signal
from core.knowledge.ranking.reliability import apply_reliability_scores
from core.knowledge.ranking.stopping import adaptive_stop_reason
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_SCIENTIFIC_EVIDENCE


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
    trial_signals: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = sanitize_api_query(ctx.semantic_query or ctx.query)
    scored, rejected = score_rows(
        rows,
        query=query,
        query_vector=ctx.query_vector,
        embed_fn=ctx.embed_fn,
        min_score=0.12,
        trial_signals=trial_signals,
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
        plan = plan_scientific_query(
            ctx.query,
            semantic_query=ctx.semantic_query or ctx.query,
        )
        query = plan.semantic_query
        semantic = plan.semantic_query
        budget = ctx.budget.max_results
        adapter_ids = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query=ctx.query,
            composer_adapter_filter=ctx.adapter_filter,
            stored_preferences=get_knowledge_source_preferences(),
        )
        discipline_match = detect_scientific_discipline(ctx.query)
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
                futures = {}
                for aid in adapter_ids:
                    search_fn = get_search_function(aid)
                    if search_fn is None:
                        continue
                    futures[
                        pool.submit(
                            search_fn,
                            adapter_query_for(plan, aid),
                            max_results=per_adapter,
                        )
                    ] = aid
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
        trial_signals = extract_trial_signal(ctx.query)
        kept, rejected = _rank_candidates(
            candidates,
            ctx=ranked_ctx,
            max_results=budget,
            trial_signals=trial_signals,
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
                {"scientific_relevance_dropped": len(rejected), "scientific_adapters_selected": list(adapter_ids), "scientific_discipline": discipline_match.discipline, "scientific_discipline_ui_group": discipline_match.ui_group},
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
            medical_query=is_medical_query(query),
        )
        conflicts = detect_conflicts(bundle.sources, topic=query)
        if conflicts:
            bundle = _bundle_with_conflicts(bundle, conflicts)
        rel_diag = {
            "scientific_relevance_dropped": len(rejected),
            "scientific_avg_relevance": round(avg_rel, 4),
            "scientific_cache_hit": cached is not None,
            "scientific_keyword_query": plan.keyword_query,
            "scientific_entity_keywords": list(plan.entity_keywords),
            "scientific_trial_signals": sorted(trial_signals),
            "scientific_adapters_selected": list(adapter_ids),
            "scientific_discipline": discipline_match.discipline,
            "scientific_discipline_ui_group": discipline_match.ui_group,
            "scientific_discipline_scores": discipline_match.scores,
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
