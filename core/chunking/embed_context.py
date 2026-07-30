"""Contextual prefix helpers for embedding (body stored separately in LanceDB)."""

from __future__ import annotations

import re

_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.+)$")


def heading_from_chunk(chunk: str) -> str | None:
    """Return the first markdown heading line in a chunk, if any."""
    for line in (chunk or "").splitlines():
        match = _HEADING_LINE_RE.match(line.strip())
        if match:
            return match.group(1).strip()
    return None


def library_chunk_embed_text(
    source: str,
    chunk: str,
    *,
    section_heading: str | None = None,
    breadcrumb: str | None = None,
) -> str:
    """
    Build the string passed to the embedder for a Library chunk.

    The raw ``chunk`` is stored in LanceDB ``text`` for UI/citations; only the
    embedder sees this prefixed form (mirrors ``help_chunk_embed_text``).
    """
    body = (chunk or "").strip()
    if not body:
        return (source or "").strip()

    heading = (section_heading or heading_from_chunk(body) or "").strip()
    trail = (breadcrumb or "").strip()
    doc_label = (source or "").strip()

    prefix_lines: list[str] = []
    if doc_label:
        prefix_lines.append(f"Document: {doc_label}")
    if trail and trail != heading:
        prefix_lines.append(f"Section: {trail}")
    elif heading:
        prefix_lines.append(f"Section: {heading}")

    if not prefix_lines:
        return body
    return f"{chr(10).join(prefix_lines)}\n\n{body}"
