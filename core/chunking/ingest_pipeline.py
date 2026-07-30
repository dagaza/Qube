"""Library ingest: Document IR → ChunkRecord list."""

from __future__ import annotations

from rag.chunker import DEFAULT_CHUNK_SIZE

from core.chunking.structure_chunker import (
    DEFAULT_LIBRARY_FALLBACK_OVERLAP_RATIO,
    ChunkRecord,
    chunk_document,
    max_chars_from_token_target,
)
from core.knowledge.document.types import Document

_MIN_CHUNK_CHARS = 50


def _breadcrumb_for_section(
    sections: list,
    section_index: int,
    *,
    heading: str | None,
    level: int,
) -> str:
    stack: list[tuple[int, str]] = []
    for idx, section in enumerate(sections[: section_index + 1]):
        sec_heading = section.heading
        sec_level = int(section.level or 0)
        if idx == section_index:
            if heading:
                sec_heading = heading
                sec_level = level or sec_level
        if not sec_heading:
            continue
        while stack and stack[-1][0] >= sec_level:
            stack.pop()
        stack.append((sec_level, sec_heading))
    return " > ".join(title for _, title in stack if title)


def _page_range_for_span(
    page_spans: list[dict[str, int]],
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    pages: list[int] = []
    for span in page_spans:
        start = int(span.get("char_start") or 0)
        end = int(span.get("char_end") or 0)
        if end > char_start and start < char_end:
            pages.append(int(span.get("page") or 0))
    if not pages:
        return None, None
    return min(pages), max(pages)


def chunk_document_for_ingest(
    document: Document,
    *,
    max_chars: int | None = None,
    target_tokens: int | None = None,
    fallback_overlap_ratio: float = DEFAULT_LIBRARY_FALLBACK_OVERLAP_RATIO,
    min_chunk_chars: int = _MIN_CHUNK_CHARS,
) -> list[ChunkRecord]:
    """
    Chunk a Document for Library indexing with breadcrumbs and optional page spans.
    """
    if max_chars is None:
        max_chars = (
            max_chars_from_token_target(target_tokens)
            if target_tokens is not None
            else DEFAULT_CHUNK_SIZE
        )
    fallback_overlap = max(0, min(int(max_chars * fallback_overlap_ratio), max_chars - 1))

    section_chunks = chunk_document(
        document,
        max_section_chars=max_chars,
        min_section_chars=min(min_chunk_chars, max_chars // 4),
        fallback_overlap=fallback_overlap,
    )

    page_spans = document.structured_data.get("page_spans") or []
    records: list[ChunkRecord] = []

    for section_chunk in section_chunks:
        section = document.sections[section_chunk.source_section_index]
        section_body = (section.text or "").strip()
        breadcrumb = _breadcrumb_for_section(
            document.sections,
            section_chunk.source_section_index,
            heading=section_chunk.heading,
            level=section_chunk.level,
        )

        char_start = 0
        char_end = 0
        if page_spans and section_body:
            needle = section_chunk.text.strip()
            found = section_body.find(needle)
            char_start = found if found >= 0 else 0
            char_end = char_start + len(needle)
        page_start, page_end = _page_range_for_span(page_spans, char_start, char_end)

        body = section_chunk.text.strip()
        if section_chunk.chunk_index > 0 and len(body) <= min_chunk_chars:
            continue

        records.append(
            ChunkRecord(
                body=body,
                heading=section_chunk.heading,
                heading_level=section_chunk.level,
                breadcrumb=breadcrumb,
                section_index=section_chunk.source_section_index,
                chunk_index=section_chunk.chunk_index,
                page_start=page_start,
                page_end=page_end,
            )
        )

    if not records:
        return records

    total = len(records)
    return [
        ChunkRecord(
            body=record.body,
            heading=record.heading,
            heading_level=record.heading_level,
            breadcrumb=record.breadcrumb,
            section_index=record.section_index,
            chunk_index=record.chunk_index,
            page_start=record.page_start,
            page_end=record.page_end,
            total_chunks=total,
        )
        for record in records
    ]
