"""
Merge retrieval results from original + assistive expanded queries.
"""
from __future__ import annotations

import copy
from typing import Any

from mcp.memory_tool import MAX_MEMORY_CHARS, MAX_MEMORY_RESULTS


def _source_key(source: dict) -> str:
    mid = source.get("memory_id")
    if mid:
        return f"mem:{mid}"
    cid = source.get("chunk_id")
    if cid:
        return f"rag:{cid}"
    return f"text:{(source.get('content') or '')[:96]}"


def merge_memory_search_results(primary: dict, auxiliary: dict) -> dict:
    """Union memory sources; primary order first, then novel auxiliary hits."""
    ps = list(primary.get("memory_sources") or [])
    aux = list(auxiliary.get("memory_sources") or [])
    seen: set[str] = set()
    merged: list[dict] = []
    for src in ps + aux:
        key = _source_key(src)
        if key in seen:
            continue
        seen.add(key)
        merged.append(copy.deepcopy(src))

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
    """Union RAG sources and rebuild llm_context from merged snippets."""
    ps = list(primary.get("sources") or [])
    aux = list(auxiliary.get("sources") or [])
    seen: set[str] = set()
    merged: list[dict] = []
    for src in ps + aux:
        key = _source_key(src)
        if key in seen:
            continue
        seen.add(key)
        merged.append(copy.deepcopy(src))

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
