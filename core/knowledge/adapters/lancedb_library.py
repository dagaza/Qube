"""LanceDB library adapter for internal corpus retrieval."""

from __future__ import annotations

from typing import Any

import numpy as np

ADAPTER_ID = "lancedb_library"
RETRIEVAL_METHOD = "hybrid_vector_fts"


def search_library_chunks(
    query: str,
    query_vector: np.ndarray,
    store: Any,
    *,
    top_k: int = 5,
    source_filter: str | None = None,
    source_prefix_filter: str | None = None,
    gate_out: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run library hybrid search; return normalized chunk rows and audit copy.

    Plan S35: unscoped searches go through the cross-encoder gate (D55).
    ``gate_out``, when given, receives the search's ``sufficiency`` and
    ``relevance_gate`` diag (counts, scores and the dropped chunks' file
    names for the inspector, D2).
    """
    from core.knowledge.ranking.library_memory_shadow import PATH_LIBRARY_TOOL
    from mcp.rag_tool import rag_search

    result = rag_search(
        query,
        query_vector,
        store,
        top_k=top_k,
        source_filter=source_filter,
        source_prefix_filter=source_prefix_filter,
        shadow_path=PATH_LIBRARY_TOOL,  # S34: offered only during a worker turn
        relevance_gate=True,  # S35 (D55); scoped searches keep the floors
    )
    if gate_out is not None:
        gate_out["sufficiency"] = result.get("sufficiency")
        gate_out["relevance_gate"] = dict(result.get("relevance_gate") or {})
    sources = result.get("sources") or []
    rows: list[dict[str, Any]] = []
    for i, src in enumerate(sources, start=1):
        filename = str(src.get("filename") or src.get("source") or f"Document {i}")
        content = str(src.get("content") or "").strip()
        rows.append(
            {
                "title": filename,
                "snippet": content,
                "source": filename,
                "chunk_id": src.get("chunk_id"),
                "full_text": content if len(content) > 400 else None,
                "_library_semantic_score": src.get("semantic_score"),
            }
        )
    return rows, [dict(r) for r in rows]
