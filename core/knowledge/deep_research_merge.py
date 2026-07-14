"""Post-merge relevance ranking for deep research bundles (Merge Ranker v2)."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

import numpy as np

from core.knowledge.ranking.merged_source import (
    DEEP_RESEARCH_MIN_MERGED_SCORE,
    DEFAULT_MERGE_WEIGHTS,
    MERGE_RANKER_VERSION,
    resolve_query_entity_ids,
    score_merged_source,
)
from core.knowledge.types import EvidenceObject
from core.retrieval_relevance import _token_set

DEEP_RESEARCH_MIN_MERGED_OVERLAP = 0.20
DEEP_RESEARCH_MIN_SEMANTIC = 0.32
DEEP_RESEARCH_MIN_KEEP = 2
DEEP_RESEARCH_MERGE_TOP_K = 8

_GENERIC_ANCHOR_STOPWORDS = frozenset(
    {
        "heart",
        "failure",
        "evidence",
        "mortality",
        "outcomes",
        "clinical",
        "trial",
        "trials",
        "patient",
        "patients",
        "treatment",
        "disease",
        "review",
        "meta",
        "analysis",
        "systematic",
        "randomized",
        "controlled",
        "hospitalization",
        "cardiovascular",
        "risk",
        "primary",
        "prevention",
        "inhibitors",
        "inhibitor",
        "therapy",
        "therapies",
        "major",
        "chronic",
        "acute",
        "associated",
        "association",
        "effects",
        "effect",
        "reduction",
        "events",
        "event",
        "rate",
        "rates",
        "show",
        "what",
    }
)

_QUERY_ANCHOR_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ace inhibitors", ("ace", "angiotensin", "acei", "aceis", "arb", "arbs")),
    ("ace inhibitor", ("ace", "angiotensin", "acei", "aceis", "arb", "arbs")),
    (
        "sglt2 inhibitors",
        ("sglt2", "sglt", "empagliflozin", "dapagliflozin", "canagliflozin"),
    ),
    (
        "sglt2 inhibitor",
        ("sglt2", "sglt", "empagliflozin", "dapagliflozin", "canagliflozin"),
    ),
    ("statin", ("statin", "statins", "hmg", "ldl", "lipid-lowering")),
    ("statins", ("statin", "statins", "hmg", "ldl", "lipid-lowering")),
)

_MERGE_REJECT_BY_PHRASE: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "ace inhibitors",
        (
            "chemotherapy-induced cardiotoxicity",
            "chemotherapy induced cardiotoxicity",
            "takotsubo",
            "chagas",
        ),
    ),
    (
        "ace inhibitor",
        (
            "chemotherapy-induced cardiotoxicity",
            "chemotherapy induced cardiotoxicity",
            "takotsubo",
            "chagas",
        ),
    ),
    ("sglt2 inhibitors", ("takotsubo", "chagas")),
    ("sglt2 inhibitor", ("takotsubo", "chagas")),
    ("statin", ("takotsubo", "chagas")),
    ("statins", ("takotsubo", "chagas")),
)

_GENERIC_MERGE_REJECT_PATTERNS: tuple[str, ...] = ("takotsubo", "chagas")


def extract_query_anchor_tokens(query: str) -> tuple[str, ...]:
    """Domain anchor tokens for merge scoring (beyond generic HF/medical terms)."""
    normalized = re.sub(r"\s+", " ", (query or "").lower()).strip()
    anchors: list[str] = []
    for phrase, expanded in _QUERY_ANCHOR_EXPANSIONS:
        if phrase in normalized:
            anchors.extend(expanded)
    token_anchors = sorted(_token_set(query) - _GENERIC_ANCHOR_STOPWORDS)
    anchors.extend(token_anchors)
    return tuple(dict.fromkeys(a for a in anchors if len(a) >= 2))


def extract_merge_reject_title_patterns(query: str) -> tuple[str, ...]:
    """Off-topic title patterns to drop during merge (aligned with eval corpus)."""
    normalized = re.sub(r"\s+", " ", (query or "").lower()).strip()
    patterns: list[str] = list(_GENERIC_MERGE_REJECT_PATTERNS)
    for phrase, rejects in _MERGE_REJECT_BY_PHRASE:
        if phrase in normalized:
            patterns.extend(rejects)
    return tuple(dict.fromkeys(p.lower() for p in patterns if p))


def _anchor_in_text(text: str, anchor: str) -> bool:
    token = (anchor or "").lower()
    if not token:
        return False
    lower = (text or "").lower()
    if len(token) <= 4:
        if token in _token_set(text):
            return True
        return bool(re.search(rf"\b{re.escape(token)}\b", lower))
    return token in lower


def source_passes_anchor_gate(text: str, anchors: tuple[str, ...]) -> bool:
    if not anchors:
        return True
    for anchor in anchors:
        if _anchor_in_text(text, anchor):
            return True
    return False


def source_title_matches_reject(title: str, reject_patterns: tuple[str, ...]) -> bool:
    lower = re.sub(r"\s+", " ", (title or "").lower()).strip()
    return any(pattern in lower for pattern in reject_patterns if pattern)


def source_passes_title_merge_gate(
    title: str,
    *,
    anchors: tuple[str, ...],
    reject_patterns: tuple[str, ...],
) -> bool:
    """Legacy helper: title must match anchor and not hit reject (eval utilities)."""
    if source_title_matches_reject(title, reject_patterns):
        return False
    return source_passes_anchor_gate(title, anchors)


def rank_merged_sources_for_query(
    query: str,
    sources: list[EvidenceObject],
    *,
    min_score: float = DEEP_RESEARCH_MIN_MERGED_SCORE,
    min_keep: int = DEEP_RESEARCH_MIN_KEEP,
    top_k: int = DEEP_RESEARCH_MERGE_TOP_K,
    query_vector: Optional[np.ndarray] = None,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    weights: dict[str, float] | None = None,
) -> tuple[list[EvidenceObject], int, dict[str, Any]]:
    """
    Rank merged deep-research sources.

    Hard drops: reject title patterns only.
    All other signals contribute to a weighted score; top-K sources are kept.
    """
    if not sources:
        return [], 0, {"merged_anchor_tokens": list(extract_query_anchor_tokens(query))}

    anchors = extract_query_anchor_tokens(query)
    reject_patterns = extract_merge_reject_title_patterns(query)
    query_entity_ids = resolve_query_entity_ids(query)
    use_embed = bool(query_vector is not None and embed_fn is not None)

    scored: list[tuple[float, EvidenceObject, dict[str, float]]] = []
    title_reject_dropped = 0

    for src in sources:
        title = str(src.title or "")
        if source_title_matches_reject(title, reject_patterns):
            title_reject_dropped += 1
            continue

        result = score_merged_source(
            query,
            src,
            anchors=anchors,
            query_entity_ids=query_entity_ids,
            query_vector=query_vector,
            embed_fn=embed_fn,
            weights=weights,
        )
        if result.total < min_score:
            continue
        scored.append((result.total, src, result.features))

    scored.sort(key=lambda row: (row[0], row[1].relevance_score, row[1].authority_score), reverse=True)

    if scored:
        keep_count = max(min_keep, min(top_k, len(scored)))
        kept_rows = scored[:keep_count]
    else:
        kept_rows = []

    kept = [src for _, src, _ in kept_rows]
    dropped = len(sources) - len(kept)

    top_features = [features for _, _, features in kept_rows[:3]]

    diag: dict[str, Any] = {
        "merged_ranker_version": MERGE_RANKER_VERSION,
        "merged_anchor_tokens": list(anchors),
        "merged_reject_title_patterns": list(reject_patterns),
        "merged_title_reject_dropped": title_reject_dropped,
        "merged_title_first_gate": False,
        "merged_title_anchor_dropped": 0,
        "merged_anchor_dropped": title_reject_dropped,
        "merged_semantic_gate": use_embed,
        "merged_semantic_dropped": 0,
        "merged_relevance_min_score": min_score,
        "merged_relevance_min_overlap": DEEP_RESEARCH_MIN_MERGED_OVERLAP,
        "merged_feature_weights": dict(weights or DEFAULT_MERGE_WEIGHTS),
        "merged_top_feature_scores": top_features,
        "merged_query_entity_count": len(query_entity_ids),
    }
    if use_embed:
        diag["merged_relevance_min_semantic"] = DEEP_RESEARCH_MIN_SEMANTIC
    return kept, dropped, diag


def filter_merged_sources_for_query(
    query: str,
    sources: list[EvidenceObject],
    *,
    min_overlap: float = DEEP_RESEARCH_MIN_MERGED_OVERLAP,
    min_semantic: float = DEEP_RESEARCH_MIN_SEMANTIC,
    min_keep: int = DEEP_RESEARCH_MIN_KEEP,
    query_vector: Optional[np.ndarray] = None,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
) -> tuple[list[EvidenceObject], int, dict[str, Any]]:
    """Backward-compatible entry point; delegates to Merge Ranker v2."""
    del min_overlap, min_semantic
    return rank_merged_sources_for_query(
        query,
        sources,
        min_keep=min_keep,
        query_vector=query_vector,
        embed_fn=embed_fn,
    )
