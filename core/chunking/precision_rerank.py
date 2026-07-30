"""Pro precision retrieval — bi-encoder rerank pass for Library RAG hits."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class _Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


def _hit_vector(hit: dict) -> np.ndarray | None:
    raw = hit.get("vector")
    if raw is None:
        return None
    try:
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return vec if vec.size else None


def score_rag_hits(
    query_vector: np.ndarray,
    hits: list[dict],
    *,
    embedder: _Embedder | None = None,
) -> list[float]:
    """Return a relevance score per hit (higher is better)."""
    if not hits:
        return []

    query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(-1))
    vectors: list[np.ndarray | None] = [_hit_vector(hit) for hit in hits]
    missing_indexes = [index for index, vec in enumerate(vectors) if vec is None]

    if missing_indexes and embedder is not None:
        missing_texts = [(hits[index].get("text") or "") for index in missing_indexes]
        embedded = np.asarray(embedder.embed(missing_texts), dtype=np.float32)
        for offset, index in enumerate(missing_indexes):
            vectors[index] = embedded[offset]

    scores: list[float] = []
    for index, hit in enumerate(hits):
        vec = vectors[index]
        if vec is not None and vec.shape[0] == query.shape[0]:
            scores.append(float(np.dot(_normalize(vec), query)))
            continue
        text = str(hit.get("text") or "")
        scores.append(float(len(text.strip()) > 0))
    return scores


def apply_precision_rerank_to_rag_hits(
    query_vector: np.ndarray,
    hits: list[dict],
    *,
    embedder: _Embedder | None = None,
) -> list[dict]:
    """Reorder hits by bi-encoder query–chunk similarity."""
    if len(hits) < 2:
        return hits
    active_embedder = embedder if embedder is not None else _lazy_rerank_embedder()
    scores = score_rag_hits(query_vector, hits, embedder=active_embedder)
    ranked = sorted(
        zip(hits, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [hit for hit, _score in ranked]


_rerank_embedder: Any | None = None


def _lazy_rerank_embedder() -> _Embedder | None:
    global _rerank_embedder
    if _rerank_embedder is not None:
        return _rerank_embedder
    try:
        from rag.embedder import EmbeddingModel

        _rerank_embedder = EmbeddingModel()
    except Exception:
        return None
    return _rerank_embedder
