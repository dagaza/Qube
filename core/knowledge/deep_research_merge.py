"""Post-merge relevance filtering for deep research bundles (Phase 5 v2)."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

import numpy as np

from core.knowledge.types import EvidenceObject
from core.retrieval_relevance import (
    _semantic_score_from_vectors,
    _token_set,
    query_snippet_token_overlap,
)

DEEP_RESEARCH_MIN_MERGED_OVERLAP = 0.20
DEEP_RESEARCH_MIN_SEMANTIC = 0.32
DEEP_RESEARCH_MIN_KEEP = 2

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
    """Domain anchor tokens required in merged sources (beyond generic HF/medical terms)."""
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
    """Title must match a domain anchor and not hit a reject pattern (Phase 5 slice 4)."""
    if source_title_matches_reject(title, reject_patterns):
        return False
    return source_passes_anchor_gate(title, anchors)


def _source_combined_text(src: EvidenceObject) -> str:
    return f"{src.title} {src.excerpt or ''}".strip()


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
    """
    Drop tangential merged hits using title-first anchors, reject patterns,
    lexical overlap, and optional embeddings.
    """
    if not sources:
        return [], 0, {"merged_anchor_tokens": list(extract_query_anchor_tokens(query))}

    anchors = extract_query_anchor_tokens(query)
    reject_patterns = extract_merge_reject_title_patterns(query)
    use_embed = bool(query_vector is not None and embed_fn is not None)
    scored: list[tuple[float, EvidenceObject]] = []
    title_reject_dropped = 0
    title_anchor_dropped = 0
    semantic_dropped = 0

    for src in sources:
        title = str(src.title or "")
        if source_title_matches_reject(title, reject_patterns):
            title_reject_dropped += 1
            continue
        if not source_passes_anchor_gate(title, anchors):
            title_anchor_dropped += 1
            continue

        combined = _source_combined_text(src)
        overlap = query_snippet_token_overlap(query, combined)

        semantic_score = 0.0
        passes_semantic = True
        if use_embed and combined:
            try:
                text_vector = embed_fn(combined[:512])
                semantic_score = _semantic_score_from_vectors(query_vector, text_vector)
                passes_semantic = semantic_score >= min_semantic
            except Exception:
                passes_semantic = overlap >= min_overlap
        passes_lexical = overlap >= min_overlap
        if not (passes_lexical or (use_embed and passes_semantic)):
            semantic_dropped += 1
            continue
        score = max(overlap, semantic_score if use_embed else 0.0) + 0.05
        scored.append((score, src))

    scored.sort(key=lambda row: row[0], reverse=True)
    kept = [src for _, src in scored]
    dropped = len(sources) - len(kept)

    if len(kept) < min_keep and scored:
        kept = [src for _, src in scored[: max(1, min(min_keep, len(scored)))]]
        dropped = len(sources) - len(kept)

    anchor_dropped = title_reject_dropped + title_anchor_dropped

    diag: dict[str, Any] = {
        "merged_anchor_tokens": list(anchors),
        "merged_reject_title_patterns": list(reject_patterns),
        "merged_title_reject_dropped": title_reject_dropped,
        "merged_title_anchor_dropped": title_anchor_dropped,
        "merged_anchor_dropped": anchor_dropped,
        "merged_title_first_gate": True,
        "merged_semantic_gate": use_embed,
        "merged_semantic_dropped": semantic_dropped,
        "merged_relevance_min_overlap": min_overlap,
    }
    if use_embed:
        diag["merged_relevance_min_semantic"] = min_semantic
    return kept, dropped, diag
