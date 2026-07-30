"""Serialize and interpret persisted Library chunk metadata (``meta_json``)."""

from __future__ import annotations

import json
from typing import Any

from core.chunking.structure_chunker import ChunkRecord

_EMPTY_META = ""


def chunk_record_to_meta_json(record: ChunkRecord) -> str:
    """Encode a ``ChunkRecord`` as a compact JSON string for LanceDB."""
    if not (record.breadcrumb or record.heading):
        return _EMPTY_META

    payload: dict[str, Any] = {
        "section_index": record.section_index,
        "chunk_index": record.chunk_index,
    }
    if record.heading:
        payload["heading"] = record.heading
    if record.heading_level:
        payload["heading_level"] = record.heading_level
    if record.breadcrumb:
        payload["breadcrumb"] = record.breadcrumb
    if record.total_chunks:
        payload["total_chunks"] = record.total_chunks
    if record.page_start is not None:
        payload["page_start"] = record.page_start
    if record.page_end is not None:
        payload["page_end"] = record.page_end
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_meta_json(raw: str | dict | None) -> dict[str, Any]:
    """Parse ``meta_json`` from LanceDB into a dict (empty on failure)."""
    if isinstance(raw, dict):
        return dict(raw)
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def section_label_from_meta(meta: dict[str, Any]) -> str | None:
    """Human-readable section label for SOURCE blocks and UI."""
    breadcrumb = str(meta.get("breadcrumb") or "").strip()
    if breadcrumb:
        return breadcrumb
    heading = str(meta.get("heading") or "").strip()
    return heading or None


def format_rag_source_header(source: str, meta: dict[str, Any] | str | None) -> str:
    """
    Format the SOURCE block header shown to the LLM.

    Example: ``manual.pdf — § Chapter 4 > Installation``
    """
    label = (source or "Unknown Document").strip()
    parsed = parse_meta_json(meta) if not isinstance(meta, dict) else meta
    section = section_label_from_meta(parsed)
    if section:
        return f"{label} — § {section}"
    return label
