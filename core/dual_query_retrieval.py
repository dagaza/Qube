"""
Merge retrieval results from original + assistive expanded queries.

Uses rank-based reciprocal fusion (``core.retrieval_fusion.fuse_ranked_results``)
so a hit strongly ranked in either query list can outrank weak primary-only rows.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from mcp.memory_tool import MAX_MEMORY_CHARS, MAX_MEMORY_RESULTS

# Primary user query list vs sidecar-expanded query list.
_PRIMARY_QUERY_WEIGHT = 1.0
_AUXILIARY_QUERY_WEIGHT = 1.0


def _source_key(source: dict) -> str:
    mid = source.get("memory_id")
    if mid:
        return f"mem:{mid}"
    cid = source.get("chunk_id")
    if cid:
        return f"rag:{cid}"
    return f"text:{(source.get('content') or '')[:96]}"


def _web_result_key(item: dict) -> str:
    url = str(item.get("url") or "").strip()
    if url:
        return f"url:{url}"
    title = str(item.get("title") or "").strip()
    snippet = str(item.get("snippet") or "").strip()[:120]
    return f"web:{title}:{snippet[:96]}"


def _merge_ranked_sources(
    primary: list[dict],
    auxiliary: list[dict],
    *,
    doc_id_fn: Callable[[dict], str],
    primary_weight: float = _PRIMARY_QUERY_WEIGHT,
    auxiliary_weight: float = _AUXILIARY_QUERY_WEIGHT,
) -> list[dict]:
    """RRF-merge two ranked source lists; dedupe by ``doc_id_fn``."""
    if not auxiliary:
        return [copy.deepcopy(s) for s in primary]
    if not primary:
        return [copy.deepcopy(s) for s in auxiliary]

    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    best_rank: dict[str, int] = {}
    primary_index: dict[str, int] = {}

    def _add(results: list[dict], weight: float) -> None:
        for rank, doc in enumerate(results):
            if not doc:
                continue
            key = doc_id_fn(doc)
            doc_map.setdefault(key, doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + weight / (rank + 1)
            best_rank[key] = min(best_rank.get(key, rank), rank)

    _add(primary, primary_weight)
    for rank, doc in enumerate(primary):
        if doc:
            primary_index.setdefault(doc_id_fn(doc), rank)
    _add(auxiliary, auxiliary_weight)

    ordered_keys = sorted(
        fused_scores.keys(),
        key=lambda k: (
            -fused_scores[k],
            best_rank[k],
            primary_index.get(k, 999),
        ),
    )
    return [copy.deepcopy(doc_map[k]) for k in ordered_keys]


def merge_memory_search_results(primary: dict, auxiliary: dict) -> dict:
    """Fuse memory sources from primary + expanded queries; rebuild context."""
    ps = list(primary.get("memory_sources") or [])
    aux = list(auxiliary.get("memory_sources") or [])
    merged = _merge_ranked_sources(ps, aux, doc_id_fn=_source_key)
    merged = merged[:MAX_MEMORY_RESULTS]

    blocks: list[str] = []
    chars = 0
    for i, src in enumerate(merged, start=1):
        src["id"] = i
        body = (src.get("content") or "").strip()
        if not body:
            continue
        line = f"- {body}"
        if chars + len(line) > MAX_MEMORY_CHARS:
            break
        chars += len(line)
        blocks.append(line)

    return {
        "memory_context": "\n".join(blocks),
        "memory_sources": merged,
    }


def merge_rag_search_results(primary: dict, auxiliary: dict) -> dict:
    """Fuse RAG sources from primary + expanded queries; rebuild llm_context."""
    ps = list(primary.get("sources") or [])
    aux = list(auxiliary.get("sources") or [])
    merged = _merge_ranked_sources(ps, aux, doc_id_fn=_source_key)

    parts: list[str] = []
    for i, src in enumerate(merged, start=1):
        src["id"] = i
        fname = src.get("filename") or "Document"
        content = (src.get("content") or "").strip()
        if content:
            parts.append(f"--- SOURCE {i}: {fname} ---\n{content}")

    return {
        "llm_context": "\n\n".join(parts),
        "sources": merged,
    }


def merge_web_search_results(
    primary: list[dict[str, Any]],
    auxiliary: list[dict[str, Any]],
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """
    Fuse web snippet rows from dual-query search (for future PR3 web hybrid path).

    Dedupes by URL when present, else title+snippet prefix.
    """
    merged = _merge_ranked_sources(
        list(primary or []),
        list(auxiliary or []),
        doc_id_fn=_web_result_key,
    )
    return merged[: max(0, int(max_results))]
