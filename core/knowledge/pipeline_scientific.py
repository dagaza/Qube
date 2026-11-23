"""Scientific evidence pipeline: PubMed + OpenAlex + arXiv with Phase 3 ranking."""

from __future__ import annotations

import dataclasses
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.adapters.registry import get_search_function
from core.app_settings import get_knowledge_source_preferences
from core.knowledge.bundle_builder import build_empty_bundle, build_scientific_evidence_bundle
from core.knowledge.conflicts.detect import detect_conflicts
from core.knowledge.scientific_adapters import is_medical_query
from core.knowledge.scientific_adapters import discipline_ordered_adapters
from core.knowledge.scientific_discipline import (
    SCIENTIFIC_DISCIPLINE_CHEMISTRY,
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_PHYSICS,
    detect_scientific_discipline,
)
from core.knowledge.scientific_query_planner import adapter_query_for, plan_scientific_query
from core.knowledge.source_preferences import (
    get_effective_enabled_adapters,
    resolve_service_adapters,
)
from core.knowledge.tiered_scientific_retrieval import (
    split_adapter_tiers,
    tiered_fallback_threshold,
    tiered_retrieval_diag,
    tiered_scientific_retrieval_enabled,
)
from core.knowledge.scientific_query_type import query_type_routing_diag
from core.knowledge.evidence_cache import get_cached_rows, make_cache_key, set_cached_rows
from core.knowledge.http_metrics import begin_turn_http_metrics, snapshot_turn_http_summary
from core.knowledge.ranking.diversity import mmr_select_rows
from core.knowledge.ranking.relevance import score_rows
from core.knowledge.ranking.trial_grounding import extract_trial_signal
from core.knowledge.ranking.reliability import apply_reliability_scores
from core.knowledge.ranking.stopping import adaptive_stop_reason
from core.knowledge.ranking.sufficiency import (
    SCIENTIFIC_THRESHOLDS,
    Sufficiency,
    SufficiencyResult,
    SufficiencyThresholds,
    assess_sufficiency,
    relevance_probability,
)
from core.knowledge.retrieval_profiles import (
    get_profile_spec,
    normalize_profile_id,
    order_adapter_ids,
    scientific_cache_enabled,
)
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


# Low-core machines score fewer rows: 40 × 800 chars took p95 1.12 s at 2 threads
# vs 0.67 s at 4 (S3 benchmark); 24 rows keeps the gate under ~0.7 s there.
_GATE_MAX_ROWS_LOW_CORE = 24
_GATE_DROPPED_DIAG_LIMIT = 40


@dataclasses.dataclass
class ScientificRanking:
    """Result of ``_rank_candidates_gated`` (plan S4)."""

    kept: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    sufficiency: SufficiencyResult
    # Rows a discipline slot may reserve: the whole pool when UNSCORED, only rows
    # that passed ``keep`` when scored.
    slot_pool: list[dict[str, Any]]
    # Rows the gate dropped (copies carrying ``_rerank_score``), best first.
    dropped: list[dict[str, Any]]
    rerank_diag: dict[str, Any]

    @property
    def scored(self) -> bool:
        return self.sufficiency.state is not Sufficiency.UNSCORED


def _gate_max_rows() -> int:
    from core.knowledge.ranking.rerank import DEFAULT_MAX_ROWS, reranker_threads

    return DEFAULT_MAX_ROWS if reranker_threads() >= 4 else _GATE_MAX_ROWS_LOW_CORE


def _rank_candidates_gated(
    rows: list[dict[str, Any]],
    *,
    ctx: RetrievalContext,
    max_results: int,
    trial_signals: frozenset[str] | None = None,
    question: str | None = None,
    thresholds: SufficiencyThresholds = SCIENTIFIC_THRESHOLDS,
) -> ScientificRanking:
    """Relevance-gated selection (S4): order by cross-encoder score, drop rows
    below ``thresholds.keep``, MMR for diversity among survivors, reliability.

    When the reranker cannot run (missing, disabled, non-English or
    content-free question, scoring error) this is exactly ``_rank_candidates``
    with ``Sufficiency.UNSCORED``: no silent weakening of today's path.
    """
    from core.knowledge.ranking import rerank

    q = str(question or ctx.semantic_query or ctx.query or "")
    diag: dict[str, Any] = {
        "model": rerank.RERANKER_MODEL_ID,
        "mode": "gate",
        "question": q[:300],
        "pool_size": len(rows),
        "threshold_keep": thresholds.keep,
        "threshold_strong": thresholds.strong,
    }

    def _legacy(reason: str) -> ScientificRanking:
        kept, rejected = _rank_candidates(
            rows, ctx=ctx, max_results=max_results, trial_signals=trial_signals
        )
        diag["skipped_reason"] = reason
        return ScientificRanking(
            kept=kept,
            rejected=rejected,
            sufficiency=assess_sufficiency(None, thresholds),
            slot_pool=list(rows),
            dropped=[],
            rerank_diag=diag,
        )

    reason = rerank.unavailable_reason(q)
    if reason is None and not rows:
        reason = rerank.SKIP_EMPTY
    if reason is not None:
        return _legacy(reason)

    # Pre-order by the legacy scorer so a row cap keeps the most promising rows,
    # and so ``_token_relevance`` stays available for diagnostics.
    token_kept, token_rejected = score_rows(
        rows,
        query=sanitize_api_query(ctx.semantic_query or ctx.query),
        query_vector=ctx.query_vector,
        embed_fn=ctx.embed_fn,
        min_score=0.12,
        trial_signals=trial_signals,
    )
    token_rejected.sort(key=lambda r: float(r.get("_scientific_relevance") or 0.0), reverse=True)
    ordered = [*token_kept, *token_rejected]
    cap = _gate_max_rows()
    to_score, unscored = ordered[:cap], ordered[cap:]
    result = rerank.rerank_rows(q, to_score, max_rows=cap)
    if result is None:
        return _legacy(rerank.SKIP_ERROR)

    for row, score in zip(to_score, result.scores):
        row["_token_relevance"] = row.get("_scientific_relevance")
        row["_rerank_score"] = round(float(score), 4)
        # One meaning everywhere (MMR, bundle, slots, UI): p(relevant).
        row["_scientific_relevance"] = relevance_probability(score)

    pool_result = assess_sufficiency(result.scores, thresholds)
    kept_idx = set(pool_result.kept_indices)
    survivors = [to_score[i] for i in pool_result.kept_indices]
    dropped = sorted(
        (r for i, r in enumerate(to_score) if i not in kept_idx),
        key=lambda r: float(r["_rerank_score"]),
        reverse=True,
    )
    # Calibrated probabilities: no min-max stretching (S4 eval: recall 0.83 → 0.86
    # at equal precision vs normalised MMR).
    selected = apply_reliability_scores(
        mmr_select_rows(survivors, max_results=max_results, normalize=False)
    )
    final = _final_sufficiency(selected, pool_result, thresholds)
    ranked_scores = sorted((float(s) for s in result.scores), reverse=True)
    diag.update(
        {
            "latency_ms": round(result.latency_ms, 1),
            "scored": len(result.scores),
            "unscored": len(unscored),
            "text_chars": result.text_chars,
            "top_scores": [round(s, 3) for s in ranked_scores[:5]],
            "kept_scores": [r.get("_rerank_score") for r in selected],
            "survivors": len(survivors),
            "dropped": len(dropped),
            "state": final.state.value,
            "strong_count": final.strong_count,
        }
    )
    return ScientificRanking(
        kept=selected,
        rejected=[*dropped, *unscored],
        sufficiency=final,
        slot_pool=survivors,
        dropped=dropped,
        rerank_diag=diag,
    )


def _final_sufficiency(
    kept: list[dict[str, Any]],
    pool_result: SufficiencyResult,
    thresholds: SufficiencyThresholds,
) -> SufficiencyResult:
    """Sufficiency of what is actually kept (after MMR / budget / slot reservation)."""
    if not kept:
        return pool_result
    scores = [r.get("_rerank_score") for r in kept]
    if any(s is None for s in scores):
        return pool_result
    return assess_sufficiency([float(s) for s in scores], thresholds)


def _dropped_for_diag(dropped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """D2: gate-dropped sources for the retrieval inspector (trace only, never the prompt)."""
    out: list[dict[str, Any]] = []
    for row in dropped[:_GATE_DROPPED_DIAG_LIMIT]:
        out.append(
            {
                "title": str(row.get("title") or "")[:200],
                "url": str(row.get("url") or "") or None,
                "adapter": str(row.get("_adapter") or "") or None,
                "score": row.get("_rerank_score"),
            }
        )
    return out


def _ensure_chemistry_pubchem_slot(
    kept: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    discipline: str,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Reserve at least one PubChem hit for chemistry discipline queries."""
    if discipline != SCIENTIFIC_DISCIPLINE_CHEMISTRY:
        return kept
    if any(str(r.get("_adapter") or "") == "pubchem" for r in kept):
        return kept
    pubchem_rows = [r for r in candidates if str(r.get("_adapter") or "") == "pubchem"]
    if not pubchem_rows:
        return kept
    best = max(
        pubchem_rows,
        key=lambda row: float(row.get("_scientific_relevance") or 0.0),
    )
    if best.get("_scientific_relevance") is None:
        from core.knowledge.ranking.relevance import score_evidence_row

        best = dict(best)
        best["_scientific_relevance"] = score_evidence_row(best, query=query)
    trimmed = kept[: max(0, max_results - 1)]
    return [best, *trimmed]


def _ensure_economics_repec_slot(
    kept: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    discipline: str,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Reserve at least one RePEc hit for economics discipline queries."""
    if discipline != SCIENTIFIC_DISCIPLINE_ECONOMICS:
        return kept
    if any(str(r.get("_adapter") or "") == "repec" for r in kept):
        return kept
    repec_rows = [r for r in candidates if str(r.get("_adapter") or "") == "repec"]
    if not repec_rows:
        return kept
    best = max(
        repec_rows,
        key=lambda row: float(row.get("_scientific_relevance") or 0.0),
    )
    if best.get("_scientific_relevance") is None:
        from core.knowledge.ranking.relevance import score_evidence_row

        best = dict(best)
        best["_scientific_relevance"] = score_evidence_row(best, query=query)
    trimmed = kept[: max(0, max_results - 1)]
    return [best, *trimmed]


def _ensure_physics_arxiv_slot(
    kept: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    discipline: str,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Reserve at least one arXiv hit for physics discipline queries."""
    if discipline != SCIENTIFIC_DISCIPLINE_PHYSICS:
        return kept
    if any(str(r.get("_adapter") or "") == "arxiv" for r in kept):
        return kept
    arxiv_rows = [r for r in candidates if str(r.get("_adapter") or "") == "arxiv"]
    if not arxiv_rows:
        return kept
    best = max(
        arxiv_rows,
        key=lambda row: float(row.get("_scientific_relevance") or 0.0),
    )
    if best.get("_scientific_relevance") is None:
        from core.knowledge.ranking.relevance import score_evidence_row

        best = dict(best)
        best["_scientific_relevance"] = score_evidence_row(best, query=query)
    trimmed = kept[: max(0, max_results - 1)]
    return [best, *trimmed]


def _collect_adapter_rows(
    adapter_ids: tuple[str, ...],
    *,
    plan: Any,
    per_adapter: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run search functions for ``adapter_ids`` in parallel."""
    adapter_calls: list[str] = []
    raw_audit: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    if not adapter_ids:
        return adapter_calls, raw_audit, candidates

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
    return adapter_calls, raw_audit, candidates


class ScientificEvidencePipeline:
    """Parallel scientific adapter collection, Phase 3 rank, and bundle assembly."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        begin_turn_http_metrics()
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
        profile = get_profile_spec(normalize_profile_id(ctx.retrieval_profile))
        adapter_ids = list(
            order_adapter_ids(tuple(adapter_ids), profile=profile)
        )
        discipline_match = detect_scientific_discipline(ctx.query)
        discipline_adapter_order = discipline_ordered_adapters(
            get_effective_enabled_adapters(
                SERVICE_SCIENTIFIC_EVIDENCE,
                stored_preferences=get_knowledge_source_preferences(),
            ),
            query=ctx.query,
        )
        query_type_diag = query_type_routing_diag(
            query=ctx.query,
            adapter_ids_before=discipline_adapter_order,
            adapter_ids_after=adapter_ids,
        )
        adapter_calls: list[str] = []
        raw_audit: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        tiered_enabled = tiered_scientific_retrieval_enabled()
        tier_primary: tuple[str, ...] = ()
        tier_fallback: tuple[str, ...] = ()
        tier_threshold = tiered_fallback_threshold(budget=budget)
        phase2_invoked = False
        candidates_after_phase1 = 0

        cache_key = make_cache_key(
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            query=semantic or query,
            adapter_filter=adapter_ids,
        )
        cached = None
        if scientific_cache_enabled(profile):
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
            if tiered_enabled:
                tier_primary, tier_fallback = split_adapter_tiers(
                    adapter_ids,
                    discipline=discipline_match.discipline,
                )
                calls1, audit1, rows1 = _collect_adapter_rows(
                    tier_primary,
                    plan=plan,
                    per_adapter=per_adapter,
                )
                adapter_calls.extend(calls1)
                raw_audit.extend(audit1)
                candidates.extend(rows1)
                candidates_after_phase1 = len(candidates)
                if len(candidates) < tier_threshold and tier_fallback:
                    phase2_invoked = True
                    calls2, audit2, rows2 = _collect_adapter_rows(
                        tier_fallback,
                        plan=plan,
                        per_adapter=per_adapter,
                    )
                    adapter_calls.extend(calls2)
                    raw_audit.extend(audit2)
                    candidates.extend(rows2)
            else:
                calls, audit, rows = _collect_adapter_rows(
                    adapter_ids,
                    plan=plan,
                    per_adapter=per_adapter,
                )
                adapter_calls.extend(calls)
                raw_audit.extend(audit)
                candidates.extend(rows)
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
        ranking = _rank_candidates_gated(
            candidates,
            ctx=ranked_ctx,
            max_results=budget,
            trial_signals=trial_signals,
            question=semantic or query,
        )
        kept, rejected = ranking.kept, ranking.rejected
        # Discipline slots may only reserve a row that passed the relevance gate
        # (the whole pool when UNSCORED, as before S4).
        slot_pool = ranking.slot_pool
        kept = _ensure_chemistry_pubchem_slot(
            kept,
            slot_pool,
            discipline=discipline_match.discipline,
            query=query,
            max_results=budget,
        )
        kept = _ensure_economics_repec_slot(
            kept,
            slot_pool,
            discipline=discipline_match.discipline,
            query=query,
            max_results=budget,
        )
        kept = _ensure_physics_arxiv_slot(
            kept,
            slot_pool,
            discipline=discipline_match.discipline,
            query=query,
            max_results=budget,
        )
        sufficiency = ranking.sufficiency
        if ranking.scored:
            sufficiency = _final_sufficiency(kept, ranking.sufficiency, SCIENTIFIC_THRESHOLDS)
        rerank_diag = dict(ranking.rerank_diag)
        if ranking.scored:
            rerank_diag["state"] = sufficiency.state.value
            rerank_diag["kept_scores"] = [r.get("_rerank_score") for r in kept]
        gate_diag: dict[str, Any] = {
            "rerank": rerank_diag,
            "sufficiency": sufficiency.state.value,
        }
        if ranking.dropped:
            gate_diag["rerank_dropped"] = _dropped_for_diag(ranking.dropped)
        latency_ms = (time.time() - t0) * 1000

        if not kept:
            empty_kwargs: dict[str, Any] = {}
            if ranking.scored:
                empty_kwargs = {
                    "sufficiency": Sufficiency.INSUFFICIENT.value,
                    "sufficiency_rationale": sufficiency.rationale,
                }
            rel_diag_empty = {
                **gate_diag,
                "scientific_relevance_dropped": len(rejected),
                "scientific_adapters_selected": list(adapter_ids),
                "scientific_discipline": discipline_match.discipline,
                "scientific_discipline_ui_group": discipline_match.ui_group,
                "scientific_query_type_routing": query_type_diag,
                "scientific_tiered_retrieval": tiered_retrieval_diag(
                    enabled=tiered_enabled,
                    primary=tier_primary,
                    fallback=tier_fallback,
                    phase2_invoked=phase2_invoked,
                    threshold=tier_threshold,
                    candidate_count=candidates_after_phase1,
                ),
                "http_summary": snapshot_turn_http_summary(
                    cache_hits_evidence=1 if cached is not None else 0,
                ),
            }
            return (
                build_empty_bundle(
                    query_raw=ctx.query,
                    query_resolved=semantic or query,
                    latency_ms=latency_ms,
                    rejected_count=len(rejected),
                    stop_reason="relevance_filtered",
                    knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
                    **empty_kwargs,
                ),
                rel_diag_empty,
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
            sufficiency=sufficiency.state.value,
            sufficiency_rationale=sufficiency.rationale if ranking.scored else "",
        )
        conflicts = detect_conflicts(bundle.sources, topic=query)
        if conflicts:
            bundle = _bundle_with_conflicts(bundle, conflicts)
        rel_diag = {
            **gate_diag,
            "scientific_relevance_dropped": len(rejected),
            "scientific_avg_relevance": round(avg_rel, 4),
            "scientific_cache_hit": cached is not None,
            "retrieval_profile": profile.id,
            "scientific_keyword_query": plan.keyword_query,
            "scientific_entity_keywords": list(plan.entity_keywords),
            "scientific_trial_signals": sorted(trial_signals),
            "scientific_adapters_selected": list(adapter_ids),
            "scientific_discipline": discipline_match.discipline,
            "scientific_discipline_ui_group": discipline_match.ui_group,
            "scientific_discipline_scores": discipline_match.scores,
            "scientific_query_type_routing": query_type_diag,
            "scientific_tiered_retrieval": tiered_retrieval_diag(
                enabled=tiered_enabled,
                primary=tier_primary,
                fallback=tier_fallback,
                phase2_invoked=phase2_invoked,
                threshold=tier_threshold,
                candidate_count=candidates_after_phase1,
            ),
            "http_summary": snapshot_turn_http_summary(
                cache_hits_evidence=1 if cached is not None else 0,
            ),
        }
        return bundle, rel_diag, raw_audit


def _bundle_with_conflicts(bundle: EvidenceBundle, conflicts):
    warnings = tuple(dict.fromkeys((*bundle.warnings, "material_conflict")))
    return dataclasses.replace(bundle, warnings=warnings, conflicts=conflicts)
