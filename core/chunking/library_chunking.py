"""Library ingest chunking — delegates to shared Document → ChunkRecord pipeline."""

from __future__ import annotations

from pathlib import Path
import re

from rag.chunker import DEFAULT_CHUNK_SIZE, chunk_text

from core.chunking.ingest_pipeline import chunk_document_for_ingest
from core.knowledge.document.builders.library_builder import (
    build_document_from_markdown,
    build_document_from_path,
)
from core.knowledge.document.types import Document, DocumentSection
from core.chunking.markdown_sections import split_markdown_sections

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _sections_from_markdown_parts(parts: list[str]) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    for part in parts:
        heading: str | None = None
        level = 0
        body_lines: list[str] = []
        for line in part.splitlines():
            match = _HEADING_LINE_RE.match(line.strip())
            if match and not body_lines:
                level = len(match.group(1))
                heading = match.group(2).strip()
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if not body and heading:
            body = heading
        if body or heading:
            sections.append(
                DocumentSection(heading=heading, level=level, text=part)
            )
    return sections


def chunk_markdown_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    all_heading_levels: bool = False,
) -> list[str]:
    """Chunk markdown; default H2/H3 split (help corpus), optional all-level split."""
    if all_heading_levels:
        document = build_document_from_markdown(text, source="inline.md")
    else:
        parts = split_markdown_sections(text)
        sections = _sections_from_markdown_parts(parts)
        document = Document(url="inline.md", title="inline", sections=sections)
    return [
        record.body
        for record in chunk_document_for_ingest(document, max_chars=max_chars)
    ]


def chunk_document_records(
    document: Document,
    *,
    max_chars: int = DEFAULT_CHUNK_SIZE,
) -> list:
    return chunk_document_for_ingest(document, max_chars=max_chars)


def chunk_library_path(
    path: Path,
    *,
    max_chars: int = DEFAULT_CHUNK_SIZE,
) -> list:
    """Chunk a Library file via the shared Document IR pipeline."""
    document = build_document_from_path(path)
    return chunk_document_for_ingest(document, max_chars=max_chars)


def chunk_library_text(
    text: str,
    *,
    path_suffix: str = "",
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 200,
) -> list[str]:
    """
    Chunk raw text for tests and legacy callers.

    Prefer ``chunk_library_path`` or ``chunk_document_for_ingest`` for ingest.
    """
    suffix = (path_suffix or "").lower()
    if suffix in (".md", ".markdown"):
        return chunk_markdown_text(text, max_chars=max_chars, all_heading_levels=True)
    if suffix == ".txt":
        from core.knowledge.document.builders.plain_text_sections import (
            split_plain_text_sections,
        )
        from core.knowledge.document.types import DocumentSection

        sections = [
            DocumentSection(heading=h, level=level, text=body)
            for h, level, body in split_plain_text_sections(text)
        ]
        document = Document(url="inline.txt", title="inline", sections=sections)
        return [record.body for record in chunk_document_for_ingest(document, max_chars=max_chars)]
    return chunk_text(text, chunk_size=max_chars, overlap=overlap)
