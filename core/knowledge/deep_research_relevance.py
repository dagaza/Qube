"""Topical relevance scoring for deep-research merged bundles (Phase 5 eval)."""

from __future__ import annotations

import re
from typing import Any, Sequence

from core.knowledge.types import EvidenceObject


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _title_passes_expect_tokens(title: str, expect_any_tokens: Sequence[str]) -> bool:
    if not expect_any_tokens:
        return True
    lower = _normalize_text(title)
    return any(token.lower() in lower for token in expect_any_tokens if token)


def _title_matches_reject_pattern(title: str, reject_patterns: Sequence[str]) -> bool:
    lower = _normalize_text(title)
    for pattern in reject_patterns:
        p = (pattern or "").strip().lower()
        if not p:
            continue
        if p in lower:
            return True
    return False


def source_title_is_relevant(
    title: str,
    *,
    expect_any_tokens: Sequence[str] = (),
    reject_title_patterns: Sequence[str] = (),
) -> bool:
    """True when title matches expected domain tokens and no reject pattern hits."""
    if _title_matches_reject_pattern(title, reject_title_patterns):
        return False
    return _title_passes_expect_tokens(title, expect_any_tokens)


def score_merged_bundle_relevance(
    sources: Sequence[EvidenceObject],
    *,
    expect_any_tokens: Sequence[str] = (),
    reject_title_patterns: Sequence[str] = (),
    top_n: int = 3,
    min_relevant_in_top: int = 2,
) -> dict[str, Any]:
    """
    Score topical alignment of the top-N merged sources.

    A source is relevant when it contains at least one ``expect_any_tokens`` entry
    (if any are configured) and does not match ``reject_title_patterns``.
    """
    top_n = max(1, int(top_n))
    min_relevant_in_top = max(1, min(int(min_relevant_in_top), top_n))
    examined = list(sources[:top_n])
    per_title: list[dict[str, Any]] = []
    relevant_count = 0

    for src in examined:
        title = str(src.title or "")
        relevant = source_title_is_relevant(
            title,
            expect_any_tokens=expect_any_tokens,
            reject_title_patterns=reject_title_patterns,
        )
        if relevant:
            relevant_count += 1
        per_title.append(
            {
                "title": title[:120],
                "relevant": relevant,
            }
        )

    has_criteria = bool(expect_any_tokens or reject_title_patterns)
    relevance_ok = (
        not has_criteria
        or relevant_count >= min_relevant_in_top
    )

    return {
        "relevance_ok": relevance_ok,
        "relevant_in_top": relevant_count,
        "relevance_top_n": top_n,
        "min_relevant_in_top": min_relevant_in_top,
        "examined_titles": per_title,
    }


def build_merge_relevance_diag(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Map deep-research pipeline diagnostics to retrieval_trace relevance_diag fields."""
    diag: dict[str, Any] = {}
    for key in (
        "merged_relevance_dropped",
        "merged_sources_pre_filter",
        "merged_sources_post_filter",
        "merged_anchor_tokens",
        "merged_anchor_dropped",
        "merged_title_reject_dropped",
        "merged_title_anchor_dropped",
        "merged_title_first_gate",
        "merged_reject_title_patterns",
        "merged_semantic_gate",
        "merged_semantic_dropped",
        "merged_relevance_min_overlap",
        "merged_relevance_min_semantic",
        "merged_ranker_version",
        "merged_relevance_min_score",
        "merged_feature_weights",
        "merged_top_feature_scores",
        "merged_query_entity_count",
        "decompose_mode",
    ):
        if key in diagnostics:
            diag[key] = diagnostics[key]
    if "merged_relevance_min_overlap" not in diag:
        from core.knowledge.deep_research_merge import DEEP_RESEARCH_MIN_MERGED_OVERLAP

        diag["merged_relevance_min_overlap"] = DEEP_RESEARCH_MIN_MERGED_OVERLAP
    return diag
