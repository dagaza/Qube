"""Backward-compatible re-exports for web fetch section chunking."""

from core.chunking.structure_chunker import (
    DEFAULT_MAX_SECTION_CHARS,
    DEFAULT_MIN_SECTION_CHARS,
    DEFAULT_MERGE_THRESHOLD,
    SectionChunk,
    chunk_document,
    chunk_section,
)

__all__ = [
    "DEFAULT_MAX_SECTION_CHARS",
    "DEFAULT_MIN_SECTION_CHARS",
    "DEFAULT_MERGE_THRESHOLD",
    "SectionChunk",
    "chunk_document",
    "chunk_section",
]
