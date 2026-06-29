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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run library hybrid search; return normalized chunk rows and audit copy."""
    from mcp.rag_tool import rag_search

    result = rag_search(
        query,
        query_vector,
        store,
        top_k=top_k,
        source_filter=source_filter,
    )
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
