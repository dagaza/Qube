"""Weighted relevance scoring for deep-research merged sources (Merge Ranker v2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.knowledge.entities.pipeline import (
    EntityResolutionContext,
    resolve_entity_ids,
)
from core.knowledge.ranking.relevance import score_evidence_row, token_overlap_score
from core.knowledge.types import EvidenceObject, SERVICE_SCIENTIFIC_EVIDENCE
from core.retrieval_relevance import _semantic_score_from_vectors, _token_set

MERGE_RANKER_VERSION = "2.0"

# Default feature weights (sum ≈ 1.0)
DEFAULT_MERGE_WEIGHTS: dict[str, float] = {
    "lexical": 0.22,
    "semantic": 0.18,
    "entity": 0.18,
    "anchor_title": 0.14,
    "anchor_excerpt": 0.08,
    "prior_relevance": 0.10,
    "authority": 0.10,
}

DEEP_RESEARCH_MIN_MERGED_SCORE = 0.14


@dataclass(frozen=True)
class MergedSourceScore:
    total: float
    features: dict[str, float]


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


def _anchor_score(text: str, anchors: tuple[str, ...]) -> float:
    if not anchors or not text:
        return 0.0
    return 1.0 if any(_anchor_in_text(text, a) for a in anchors) else 0.0


def _entity_overlap_score(
    query_entity_ids: tuple[str, ...],
    source_entity_ids: tuple[str, ...],
) -> float:
    if not query_entity_ids:
        return 0.0
    if not source_entity_ids:
        return 0.0
    overlap = set(query_entity_ids) & set(source_entity_ids)
    if not overlap:
        return 0.0
    return min(1.0, len(overlap) / max(1, len(query_entity_ids)))


def _combined_text(src: EvidenceObject) -> str:
    return " ".join(
        part
        for part in (
            src.title,
            src.excerpt or "",
            src.full_text or "",
        )
        if part
    ).strip()


def score_merged_source(
    query: str,
    source: EvidenceObject,
    *,
    anchors: tuple[str, ...] = (),
    query_entity_ids: tuple[str, ...] = (),
    source_entity_ids: tuple[str, ...] | None = None,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    weights: dict[str, float] | None = None,
) -> MergedSourceScore:
    """Score one merged candidate using composable features (0–1 each)."""
    w = weights or DEFAULT_MERGE_WEIGHTS
    combined = _combined_text(source)
    title = str(source.title or "")

    lexical = token_overlap_score(query, combined)
    semantic = 0.0
    if query_vector is not None and embed_fn is not None and combined:
        try:
            doc_vec = embed_fn(combined[:512])
            semantic = _semantic_score_from_vectors(query_vector, doc_vec)
            semantic = max(0.0, min(1.0, (semantic + 1.0) / 2.0))
        except Exception:
            row_score = score_evidence_row(
                {"title": title, "snippet": source.excerpt, "full_text": source.full_text},
                query=query,
            )
            semantic = row_score * 0.5
    elif combined:
        semantic = score_evidence_row(
            {"title": title, "snippet": source.excerpt, "full_text": source.full_text},
            query=query,
        ) * 0.45

    anchor_title = _anchor_score(title, anchors)
    excerpt_text = " ".join(
        part for part in (source.excerpt or "", source.full_text or "") if part
    ).strip()
    anchor_excerpt = _anchor_score(excerpt_text, anchors) if excerpt_text else 0.0
    if anchor_title >= 1.0:
        anchor_excerpt = 0.0

    sids = source_entity_ids
    if sids is None:
        sids = source.entity_ids or ()
    entity_raw = _entity_overlap_score(query_entity_ids, sids)
    if anchors:
        # Entity corroboration applies when the title already signals a query anchor.
        entity = entity_raw * anchor_title
    else:
        entity = entity_raw

    prior = max(0.0, min(1.0, float(source.relevance_score or 0.0)))
    authority = max(0.0, min(1.0, float(source.authority_score or 0.0)))

    features = {
        "lexical": round(lexical, 4),
        "semantic": round(semantic, 4),
        "entity": round(entity, 4),
        "anchor_title": round(anchor_title, 4),
        "anchor_excerpt": round(anchor_excerpt, 4),
        "prior_relevance": round(prior, 4),
        "authority": round(authority, 4),
    }

    total = sum(features[key] * w.get(key, 0.0) for key in features)
    return MergedSourceScore(total=round(max(0.0, min(1.0, total)), 4), features=features)


def resolve_query_entity_ids(
    query: str,
    *,
    knowledge_service: str = SERVICE_SCIENTIFIC_EVIDENCE,
) -> tuple[str, ...]:
    ctx = EntityResolutionContext(
        query_resolved=query,
        knowledge_service=knowledge_service,
    )
    return resolve_entity_ids(query, ctx)
