"""
Shared rank-based fusion for vector + lexical retrieval channels.

Used by ``mcp.memory_tool`` and ``mcp.rag_tool`` so hybrid merge behaviour
stays consistent across RAG and memory search paths.
"""
from __future__ import annotations

from typing import Callable, Iterable


def bm25_rank_to_score(rank: float) -> float:
    """Map a BM25 rank (lower is better) to a normalized relevance score in [0, 1].

    Higher rank indices (worse matches) decay toward zero via ``1 / (1 + rank)``.
    """
    try:
        r = max(0.0, float(rank))
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + r)


def _default_doc_id(doc: dict) -> str:
    return str(
        doc.get("id")
        or doc.get("chunk_id")
        or doc.get("source")
        or (doc.get("text") or "")[:64]
    )


def _fts_has_score_metadata(text_results: Iterable[dict]) -> bool:
    for doc in text_results or []:
        if not doc:
            continue
        if doc.get("_score") is not None or doc.get("score") is not None:
            return True
    return False


def fuse_weighted_scores(
    vector_results: Iterable[dict],
    text_results: Iterable[dict],
    *,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    doc_id_fn: Callable[[dict], str] | None = None,
) -> list[tuple[dict, set[str]]]:
    """Fuse vector distance + FTS BM25 rank when lexical scores are available."""
    id_fn = doc_id_fn or _default_doc_id
    fused: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    channels: dict[str, set[str]] = {}

    for rank, doc in enumerate(vector_results or []):
        if not doc:
            continue
        doc_id = id_fn(doc)
        distance = float(doc.get("_distance", 1.0))
        vector_score = max(0.0, 1.0 - distance)
        fused[doc_id] = fused.get(doc_id, 0.0) + vector_weight * vector_score
        doc_map.setdefault(doc_id, doc)
        channels.setdefault(doc_id, set()).add("vector")

    for rank, doc in enumerate(text_results or []):
        if not doc:
            continue
        doc_id = id_fn(doc)
        raw = doc.get("_score", doc.get("score"))
        if raw is not None:
            text_score = bm25_rank_to_score(float(raw))
        else:
            text_score = bm25_rank_to_score(float(rank))
        fused[doc_id] = fused.get(doc_id, 0.0) + text_weight * text_score
        doc_map.setdefault(doc_id, doc)
        channels.setdefault(doc_id, set()).add("fts")

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_map[doc_id], set(channels.get(doc_id, set()))) for doc_id, _ in ordered]


def fuse_ranked_results(
    vector_results: Iterable[dict],
    text_results: Iterable[dict],
    *,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    doc_id_fn: Callable[[dict], str] | None = None,
) -> list[tuple[dict, set[str]]]:
    """Merge two ranked lists with rank-based reciprocal scoring.

    Returns ``[(doc, channels), ...]`` sorted by fused score descending.
    ``channels`` is a subset of ``{"vector", "fts"}``.
    """
    id_fn = doc_id_fn or _default_doc_id
    fused: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    channels: dict[str, set[str]] = {}

    def _add(results: Iterable[dict], weight: float, channel: str) -> None:
        for rank, doc in enumerate(results):
            if not doc:
                continue
            doc_id = id_fn(doc)
            if doc_id not in fused:
                fused[doc_id] = 0.0
                doc_map[doc_id] = doc
                channels[doc_id] = set()
            fused[doc_id] += weight / (rank + 1)
            channels[doc_id].add(channel)

    _add(vector_results, vector_weight, "vector")
    _add(text_results, text_weight, "fts")

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_map[doc_id], set(channels.get(doc_id, set()))) for doc_id, _ in ordered]
