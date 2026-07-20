"""Heading-aware chunking for help corpus markdown (§3.8)."""

from __future__ import annotations

import re

from rag.chunker import chunk_text

_HEADING_SPLIT_RE = re.compile(r"(?=^#{2,3}\s+)", re.MULTILINE)
_DEFAULT_MAX_CHARS = 1500


def split_help_markdown_sections(text: str) -> list[str]:
    """Split markdown on H2/H3 boundaries, keeping heading lines with body."""
    body = (text or "").strip()
    if not body:
        return []
    parts = [part.strip() for part in _HEADING_SPLIT_RE.split(body) if part.strip()]
    if not parts:
        return [body]
    if parts[0].startswith("# ") and not parts[0].startswith("##"):
        parts = parts[1:]
    return parts


def chunk_help_markdown(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[str]:
    """
    Chunk help docs at semantic heading boundaries.

    Long sections are further split with the standard RAG chunker hard cap.
    """
    sections = split_help_markdown_sections(text)
    if not sections:
        return []

    out: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            out.append(section)
            continue
        out.extend(chunk_text(section, chunk_size=max_chars))

    cleaned = [chunk.strip() for chunk in out if chunk.strip()]
    if not cleaned and text.strip():
        return chunk_text(text.strip(), chunk_size=max_chars)
    return cleaned
