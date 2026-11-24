"""Deep research orchestration (Phase 4): decompose, retrieve, merge, report."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from core.knowledge.entities.enrich import enrich_evidence_object
from core.knowledge.entities.policy import dedupe_cluster_entity_ids
from core.knowledge.entities.types import EntityResolutionContext
from core.knowledge.conflicts.detect import detect_conflicts
from core.knowledge.types import (
    COVERAGE_ADEQUATE,
    COVERAGE_EXCELLENT,
    COVERAGE_NONE,
    COVERAGE_POOR,
    EvidenceBundle,
    EvidenceObject,
    RetrievalBudget,
    SERVICE_SCIENTIFIC_EVIDENCE,
    SERVICE_TRUSTED_KNOWLEDGE,
)
from core.knowledge.deep_research_decompose import decompose_query, normalize_deep_research_query
from core.knowledge.deep_research_merge import DEEP_RESEARCH_MERGE_TOP_K, filter_merged_sources_for_query
from core.knowledge.deep_research_profiles import (
    DEEP_BUDGET,
    DeepResearchProfileSpec,
    get_profile_spec,
)
from core.knowledge.web_retrieval import run_v2_web_retrieval

DEEP_RESEARCH_PROFILE_VERSION = "0.2.0"
DEEP_RESEARCH_STRATEGY = "deep_research_merged"


class DeepResearchCancelled(Exception):
    """Raised when a deep-research job is cancelled mid-pipeline."""


@dataclass(frozen=True)
class DeepResearchProgress:
    phase: str
    message: str
    sub_query_index: int = 0
    sub_query_total: int = 0
    sources_found: int = 0
    sub_queries: tuple[str, ...] = ()


@dataclass
class DeepResearchResult:
    query: str
    sub_queries: tuple[str, ...]
    merged_bundle: EvidenceBundle | None
    sub_bundles: tuple[EvidenceBundle, ...] = ()
    report_markdown: str = ""
    latency_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _normalize_title(title: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _dedupe_sources(
    sources: list[EvidenceObject],
    *,
    ctx: EntityResolutionContext | None = None,
) -> list[EvidenceObject]:
    """Dedupe cross sub-query hits; prefer DOI → entity cluster → title → URL."""
    best: dict[str, EvidenceObject] = {}
    order: list[str] = []

    def _dedupe_key(src: EvidenceObject) -> str:
        doi = (src.doi or "").strip().lower()
        if doi:
            return f"doi:{doi}"

        enriched = enrich_evidence_object(src, ctx)
        cluster = dedupe_cluster_entity_ids(enriched.entity_ids)
        if cluster:
            return f"cluster:{'|'.join(cluster)}"

        title_key = _normalize_title(src.title)
        if title_key:
            return f"title:{title_key}"
        url = (src.url or "").strip().lower().rstrip("/")
        if url:
            return f"url:{url}"
        sid = (src.source_id or src.id or "").strip()
        return f"id:{src.adapter}:{sid}"

    for src in sources:
        key = _dedupe_key(src)
        prev = best.get(key)
        if prev is None:
            best[key] = src
            order.append(key)
            continue
        if (src.relevance_score, src.authority_score) > (
            prev.relevance_score,
            prev.authority_score,
        ):
            best[key] = src
    return [best[k] for k in order]


def merge_evidence_bundles(
    *,
    query: str,
    bundles: tuple[EvidenceBundle, ...],
    knowledge_service: str = SERVICE_SCIENTIFIC_EVIDENCE,
    merged_source_cap: int | None = None,
    profile: DeepResearchProfileSpec | None = None,
) -> EvidenceBundle | None:
    """Merge sub-query bundles into one ranked evidence bundle."""
    nonempty = [b for b in bundles if b.sources]
    if not nonempty:
        return None

    resolution_ctx = EntityResolutionContext(
        query_resolved=normalize_deep_research_query(query),
        knowledge_service=knowledge_service,
    )

    merged_sources = _dedupe_sources(
        [enrich_evidence_object(src, resolution_ctx) for bundle in nonempty for src in bundle.sources],
        ctx=resolution_ctx,
    )
    merged_sources.sort(
        key=lambda s: (s.relevance_score, s.authority_score),
        reverse=True,
    )
    if merged_source_cap is None:
        spec = profile or get_profile_spec(None)
        merged_source_cap = spec.merged_source_cap
    capped = tuple(merged_sources[: max(1, merged_source_cap)])

    abstract_count = sum(1 for s in capped if s.fetch_status == "abstract")
    adapters = {s.adapter for s in capped}
    if len(capped) >= 4 and abstract_count >= 2 and len(adapters) >= 2:
        coverage, coverage_rationale = (
            COVERAGE_EXCELLENT,
            f"{len(capped)} merged sources across {len(adapters)} indexes; "
            f"{abstract_count} abstracts.",
        )
    elif len(capped) >= 2 and abstract_count >= 1:
        coverage, coverage_rationale = (
            COVERAGE_ADEQUATE,
            f"{len(capped)} merged source(s); {abstract_count} with abstracts.",
        )
    elif abstract_count >= 1:
        coverage, coverage_rationale = (
            COVERAGE_POOR,
            "Limited merged coverage; corroboration may be insufficient.",
        )
    else:
        coverage, coverage_rationale = (
            COVERAGE_POOR,
            "Merged hits lacked abstracts; snippet-only coverage.",
        )

    confidence = (
        sum(s.relevance_score * 0.55 + s.authority_score * 0.45 for s in capped)
        / len(capped)
        if capped
        else 0.0
    )
    authority_summary = (
        sum(s.authority_score for s in capped) / len(capped) if capped else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in capped) / len(capped) if capped else 0.0
    )
    diversity_summary = min(1.0, len(adapters) / 3.0)
    warnings = tuple(
        dict.fromkeys(w for bundle in nonempty for w in bundle.warnings)
    )
    adapter_calls = tuple(
        sorted({a for bundle in nonempty for a in bundle.adapter_calls})
    )
    rejected = sum(bundle.rejected_count for bundle in nonempty)
    latency_ms = sum(bundle.latency_ms for bundle in nonempty)
    conflicts = detect_conflicts(capped, topic=query)

    from core.knowledge.entities.enrich import enrich_bundle

    merged_bundle = EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query,
        query_resolved=normalize_deep_research_query(query),
        knowledge_service=knowledge_service,
        retrieval_strategy=DEEP_RESEARCH_STRATEGY,
        profile_version=DEEP_RESEARCH_PROFILE_VERSION,
        retrieved_at=time.time(),
        latency_ms=latency_ms,
        confidence=min(0.95, confidence),
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=capped,
        rejected_count=rejected,
        warnings=warnings,
        conflicts=conflicts,
        stop_reason="sufficient_evidence" if len(capped) >= 3 else "budget_exhausted",
        adapter_calls=adapter_calls,
    )
    return enrich_bundle(merged_bundle, resolution_ctx)


def build_bibliography_report(
    *,
    query: str,
    bundle: EvidenceBundle | None,
    sub_queries: tuple[str, ...],
    include_summary: bool = True,
    sub_query_fallbacks: tuple[dict[str, Any], ...] = (),
) -> str:
    """Markdown report skeleton with numbered bibliography (Phase 4 v1)."""
    lines = [
        "# Deep Research Report",
        "",
        f"**Query:** {query.strip()}",
        "",
    ]
    if sub_queries and len(sub_queries) > 1:
        lines.extend(["## Sub-queries", ""])
        for idx, sq in enumerate(sub_queries, start=1):
            lines.append(f"{idx}. {sq}")
        lines.append("")
    # S12 (D14): say which sub-questions had no relevant scholarly source.
    for fb in sub_query_fallbacks:
        target = "trusted sources (Wikipedia, government, intergovernmental)"
        if fb.get("sources"):
            lines.append(
                f"_Sub-query {fb.get('index')}: no relevant scholarly source; "
                f"{fb.get('sources')} result(s) from {target} were used instead._"
            )
        else:
            lines.append(
                f"_Sub-query {fb.get('index')}: no relevant scholarly or trusted source found._"
            )
    if sub_query_fallbacks:
        lines.append("")

    if bundle is None or not bundle.sources:
        lines.extend(
            [
                "## Findings",
                "",
                "_No evidence sources were retained after retrieval and ranking._",
                "",
            ]
        )
        return "\n".join(lines)

    if include_summary:
        lines.extend(
            [
                "## Summary",
                "",
                bundle.coverage_rationale,
                "",
                f"Coverage: **{bundle.coverage}** · Confidence: **{bundle.confidence:.2f}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Bibliography",
            "",
        ]
    )
    for idx, src in enumerate(bundle.sources, start=1):
        authors = ", ".join(src.authors[:3]) if src.authors else "Unknown"
        venue = f" — {src.venue}" if src.venue else ""
        date = f" ({src.publication_date})" if src.publication_date else ""
        url = f" <{src.url}>" if src.url else ""
        lines.append(
            f"{idx}. **{src.title}**{venue}{date}. {authors}.{url}"
        )
    if bundle.conflicts:
        lines.extend(["", "## Conflicts noted", ""])
        for conflict in bundle.conflicts:
            lines.append(f"- **{conflict.topic}** ({conflict.severity})")
    return "\n".join(lines) + "\n"


# S12 (R10): the merged gate's keep threshold, the scientific gate's (S4).
MERGE_RERANK_KEEP = 0.0


def rerank_merged_sources(
    query: str,
    sources: list[EvidenceObject],
    *,
    top_k: int,
) -> tuple[list[EvidenceObject], int, dict[str, Any]] | None:
    """S12: rank merged sources with the S3 cross-encoder; ``None`` if it cannot run.

    Each source scores the higher of its cross-encoder score against the main
    question and the score its own sub-question gave it (``raw_metadata
    ["rerank_score"]``), so a source relevant to one sub-question is not lost
    because the main question is broader. Sources below the scientific keep
    threshold on both are dropped; the best ``top_k`` are kept.
    """
    from core.knowledge.ranking import rerank

    if not sources or rerank.unavailable_reason(query) is not None:
        return None
    # Title + 800-char head, as every other gate scores (V5).
    texts = [
        f"{s.title}. {(s.full_text or s.excerpt or '').strip()}"[: rerank.DEFAULT_TEXT_CHARS]
        for s in sources
    ]
    result = rerank.rerank_texts(query, texts)
    if result is None:
        return None
    scored: list[tuple[float, float, EvidenceObject]] = []
    for src, main_score in zip(sources, result.scores):
        sub = (src.raw_metadata or {}).get("rerank_score")
        try:
            sub_score = float(sub) if sub is not None else float("-inf")
        except (TypeError, ValueError):
            sub_score = float("-inf")
        best = max(float(main_score), sub_score)
        scored.append((best, float(main_score), src))
    passing = [row for row in scored if row[0] >= MERGE_RERANK_KEEP]
    passing.sort(key=lambda row: (row[0], row[2].authority_score), reverse=True)
    kept_rows = passing[: max(1, int(top_k))]
    kept = [src for _, _, src in kept_rows]
    diag = {
        "merged_gate": "rerank",
        "merged_rerank_keep": MERGE_RERANK_KEEP,
        "merged_rerank_top_k": int(top_k),
        "merged_rerank_below_keep": len(scored) - len(passing),
        "merged_rerank_over_cap": max(0, len(passing) - len(kept_rows)),
        "merged_rerank_scores": [
            {"title": src.title[:80], "best": round(best, 2), "main": round(main, 2)}
            for best, main, src in kept_rows[:8]
        ],
        "merged_rerank_latency_ms": round(result.latency_ms, 1),
    }
    return kept, len(sources) - len(kept), diag


def apply_merged_relevance_gate(
    *,
    query: str,
    bundle: EvidenceBundle | None,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    top_k: int | None = None,
) -> tuple[EvidenceBundle | None, int, dict[str, Any]]:
    """Filter tangential merged sources; rebuild bundle metrics when sources drop.

    S12 (R10): the cross-encoder ranks the merge when it can run (the same
    gate as each sub-question); Merge Ranker v2 is the fallback. ``top_k`` is
    the profile's merged-source cap (it used to be a fixed 8, below both
    profiles' caps of 10 and 24).
    """
    if bundle is None or not bundle.sources:
        return bundle, 0, {}

    cap = int(top_k) if top_k else DEEP_RESEARCH_MERGE_TOP_K
    reranked = rerank_merged_sources(query, list(bundle.sources), top_k=cap)
    if reranked is not None:
        filtered, dropped, filter_diag = reranked
    else:
        filtered, dropped, filter_diag = filter_merged_sources_for_query(
            query,
            list(bundle.sources),
            query_vector=query_vector,
            embed_fn=embed_fn,
            top_k=cap,
        )
        filter_diag = {**filter_diag, "merged_gate": "merge_ranker_v2"}
    if not filtered:
        return None, dropped, filter_diag
    filtered_ids = [s.id for s in filtered]
    original_ids = [s.id for s in bundle.sources]
    if dropped <= 0 and filtered_ids == original_ids:
        return bundle, 0, filter_diag

    filtered_sources = tuple(filtered)
    adapters = {s.adapter for s in filtered_sources}
    abstract_count = sum(1 for s in filtered_sources if s.fetch_status == "abstract")
    if len(filtered_sources) >= 4 and abstract_count >= 2 and len(adapters) >= 2:
        coverage, coverage_rationale = (
            COVERAGE_EXCELLENT,
            f"{len(filtered_sources)} relevance-filtered sources across {len(adapters)} indexes; "
            f"{abstract_count} abstracts.",
        )
    elif len(filtered_sources) >= 2 and abstract_count >= 1:
        coverage, coverage_rationale = (
            COVERAGE_ADEQUATE,
            f"{len(filtered_sources)} relevance-filtered source(s); {abstract_count} with abstracts.",
        )
    elif abstract_count >= 1:
        coverage, coverage_rationale = (
            COVERAGE_POOR,
            "Limited relevance-filtered coverage; corroboration may be insufficient.",
        )
    else:
        coverage, coverage_rationale = (
            COVERAGE_POOR,
            "Relevance-filtered hits lacked abstracts; snippet-only coverage.",
        )

    confidence = (
        sum(s.relevance_score * 0.55 + s.authority_score * 0.45 for s in filtered_sources)
        / len(filtered_sources)
    )
    return (
        replace(
            bundle,
            confidence=min(0.95, confidence),
            coverage=coverage,
            coverage_rationale=coverage_rationale,
            authority_summary=(
                sum(s.authority_score for s in filtered_sources) / len(filtered_sources)
            ),
            reliability_summary=(
                sum(s.reliability_score for s in filtered_sources) / len(filtered_sources)
            ),
            diversity_summary=min(1.0, len(adapters) / 3.0),
            sources=filtered_sources,
            rejected_count=bundle.rejected_count + dropped,
            conflicts=detect_conflicts(filtered_sources, topic=query),
        ),
        dropped,
        filter_diag,
    )


def _resolve_deep_research_embed_context(
    query: str,
    *,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    query_vector: np.ndarray | None = None,
) -> tuple[Callable[[str], np.ndarray] | None, np.ndarray | None]:
    if embed_fn is not None and query_vector is not None:
        return embed_fn, query_vector
    return None, None


def run_deep_research(
    query: str,
    *,
    knowledge_service: str = SERVICE_SCIENTIFIC_EVIDENCE,
    progress_cb: Callable[[DeepResearchProgress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    query_vector: np.ndarray | None = None,
    decompose_generate_fn: Callable[[str, str], str] | None = None,
    decompose_mode: str | None = None,
    profile: DeepResearchProfileSpec | None = None,
    profile_id: str | None = None,
    force_thorough: bool = False,
) -> DeepResearchResult:
    """Sync deep-research pipeline: decompose → retrieve → merge → report."""
    from core.deep_research_pro_features import resolve_deep_research_profile

    resolved_profile = resolve_deep_research_profile(
        profile_id=profile_id,
        force_thorough=force_thorough,
    )
    profile_spec = profile or resolved_profile.spec
    t0 = time.time()
    normalized_query = normalize_deep_research_query(query)
    resolved_embed_fn, resolved_query_vector = _resolve_deep_research_embed_context(
        normalized_query,
        embed_fn=embed_fn,
        query_vector=query_vector,
    )

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise DeepResearchCancelled()

    def _emit(
        phase: str,
        message: str,
        *,
        idx: int = 0,
        total: int = 0,
        sources_found: int = 0,
        sub_queries: tuple[str, ...] = (),
    ) -> None:
        if progress_cb is not None:
            progress_cb(
                DeepResearchProgress(
                    phase=phase,
                    message=message,
                    sub_query_index=idx,
                    sub_query_total=total,
                    sources_found=sources_found,
                    sub_queries=sub_queries,
                )
            )

    _check_cancel()
    _emit("decomposing", "Planning sub-queries…")
    if decompose_mode is None:
        mode = "llm" if decompose_generate_fn is not None else "heuristic"
    else:
        mode = decompose_mode.strip().lower()
    if mode in {"llm", "hybrid"} and decompose_generate_fn is None:
        mode = "heuristic"
    sub_queries = decompose_query(
        normalized_query,
        generate_fn=decompose_generate_fn,
        mode=mode,
        max_sub_queries=profile_spec.max_sub_queries,
    )
    decompose_method = mode
    if not sub_queries:
        return DeepResearchResult(
            query=normalized_query,
            sub_queries=(),
            merged_bundle=None,
            latency_ms=(time.time() - t0) * 1000,
        )

    sub_bundles: list[EvidenceBundle] = []
    sub_query_fallbacks: list[dict[str, Any]] = []
    total = len(sub_queries)
    accumulated_sources = 0
    for idx, sub_q in enumerate(sub_queries, start=1):
        _check_cancel()
        _emit(
            "retrieving",
            f"Retrieving evidence ({idx}/{total})…",
            idx=idx,
            total=total,
            sources_found=accumulated_sources,
            sub_queries=sub_queries,
        )
        outcome = run_v2_web_retrieval(
            query=sub_q,
            semantic_query=sub_q,
            knowledge_service=knowledge_service,
            budget=profile_spec.budget,
            # S12 (D16): the @research depth profile is not capped by the app profile.
            budget_is_explicit=True,
        )
        if (
            knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE
            and outcome.bundle is not None
            and getattr(outcome.bundle, "sufficiency", "") == "insufficient"
        ):
            # S12 (D14): the relevance gate found no scholarly source for this
            # sub-question; retry it once on trusted sources.
            trusted = run_v2_web_retrieval(
                query=sub_q,
                semantic_query=sub_q,
                knowledge_service=SERVICE_TRUSTED_KNOWLEDGE,
                budget=profile_spec.budget,
                budget_is_explicit=True,
            )
            fallback_used = bool(trusted.bundle is not None and trusted.bundle.sources)
            sub_query_fallbacks.append(
                {
                    "sub_query": sub_q[:200],
                    "index": idx,
                    "to": SERVICE_TRUSTED_KNOWLEDGE,
                    "sources": len(trusted.bundle.sources) if fallback_used else 0,
                }
            )
            if fallback_used:
                # Kept next to (not instead of) any scholarly rows; the merged
                # gate ranks them together.
                sub_bundles.append(trusted.bundle)
        if outcome.bundle is not None:
            sub_bundles.append(outcome.bundle)
            accumulated_sources = sum(len(b.sources) for b in sub_bundles)
            _emit(
                "retrieving",
                f"Retrieved {accumulated_sources} source(s) so far…",
                idx=idx,
                total=total,
                sources_found=accumulated_sources,
                sub_queries=sub_queries,
            )

    _check_cancel()
    _emit("merging", "Merging and de-duplicating sources…")
    merged = merge_evidence_bundles(
        query=normalized_query,
        bundles=tuple(sub_bundles),
        knowledge_service=knowledge_service,
        profile=profile_spec,
    )
    pre_filter_count = len(merged.sources) if merged else 0
    merged, dropped, filter_diag = apply_merged_relevance_gate(
        query=normalized_query,
        bundle=merged,
        query_vector=resolved_query_vector,
        embed_fn=resolved_embed_fn,
        top_k=profile_spec.merged_source_cap,
    )
    post_filter_count = len(merged.sources) if merged else 0
    _check_cancel()
    _emit("reporting", "Building bibliography…")
    report = build_bibliography_report(
        query=normalized_query,
        bundle=merged,
        sub_queries=sub_queries,
        sub_query_fallbacks=tuple(sub_query_fallbacks),
    )
    latency_ms = (time.time() - t0) * 1000
    return DeepResearchResult(
        query=normalized_query,
        sub_queries=sub_queries,
        merged_bundle=merged,
        sub_bundles=tuple(sub_bundles),
        report_markdown=report,
        latency_ms=latency_ms,
        diagnostics={
            "sub_query_count": len(sub_queries),
            "sub_bundle_count": len(sub_bundles),
            "decompose_method": decompose_method,
            "decompose_mode": mode,
            "deep_research_profile": profile_spec.id,
            "deep_research_profile_requested": resolved_profile.requested_id,
            "deep_research_profile_downgraded": resolved_profile.downgraded,
            "merged_source_count": post_filter_count,
            "merged_sources_pre_filter": pre_filter_count,
            "merged_sources_post_filter": post_filter_count,
            "merged_relevance_dropped": dropped,
            "sub_query_fallbacks": sub_query_fallbacks,
            **filter_diag,
        },
    )
