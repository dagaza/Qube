"""Shared structure-aware chunking for Library ingest and web fetch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.chunker import chunk_text

if TYPE_CHECKING:
    from core.knowledge.document.types import Document, DocumentSection

# Web fetch defaults (backward compatible with legacy section_chunker).
DEFAULT_MAX_SECTION_CHARS = 800
DEFAULT_MIN_SECTION_CHARS = 200
DEFAULT_MERGE_THRESHOLD = 150

# Library ingest defaults (character cap; token heuristic in ingest_pipeline).
DEFAULT_LIBRARY_MAX_CHARS = 1500
DEFAULT_LIBRARY_FALLBACK_OVERLAP_RATIO = 0.10

# Heuristic: Latin prose ≈ 4 characters per token (Phase 2 v1).
CHARS_PER_TOKEN_HEURISTIC = 4
DEFAULT_TARGET_TOKENS = 512

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SectionChunk:
    heading: str | None
    level: int
    text: str
    source_section_index: int
    chunk_index: int = 0


@dataclass(frozen=True)
class ChunkRecord:
    """Library ingest chunk with structural metadata."""

    body: str
    heading: str | None
    heading_level: int
    breadcrumb: str
    section_index: int
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    total_chunks: int = 0

    @property
    def embed_text(self) -> str:
        return self.body


def estimate_tokens(text: str) -> int:
    """Embedder-agnostic token estimate for sizing heuristics."""
    return max(1, len(text or "") // CHARS_PER_TOKEN_HEURISTIC)


def max_chars_from_token_target(
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    *,
    hard_cap: int = DEFAULT_LIBRARY_MAX_CHARS,
) -> int:
    """Convert a token target to a character budget with a hard cap."""
    return min(hard_cap, max(1, target_tokens * CHARS_PER_TOKEN_HEURISTIC))


def _section_body(section: DocumentSection) -> str:
    parts: list[str] = []
    if section.text:
        parts.append(section.text.strip())
    if section.list_items:
        parts.extend(f"- {item.strip()}" for item in section.list_items if item.strip())
    return "\n".join(parts).strip()


def _split_sentences(
    text: str,
    *,
    max_chars: int,
    fallback_overlap: int = 0,
) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text[:max_chars].strip()] if text else []

    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            if fallback_overlap > 0:
                chunks.extend(
                    chunk_text(
                        sentence,
                        chunk_size=max_chars,
                        overlap=fallback_overlap,
                    )
                )
            else:
                for start in range(0, len(sentence), max_chars):
                    piece = sentence[start : start + max_chars].strip()
                    if piece:
                        chunks.append(piece)
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def _split_paragraphs(
    text: str,
    *,
    max_chars: int,
    fallback_overlap: int = 0,
) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return _split_sentences(
            text,
            max_chars=max_chars,
            fallback_overlap=fallback_overlap,
        )

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(
                _split_sentences(
                    paragraph,
                    max_chars=max_chars,
                    fallback_overlap=fallback_overlap,
                )
            )
            continue
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


def _merge_small_chunks(
    chunks: list[str],
    *,
    max_chars: int,
    merge_threshold: int,
) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = []
    pending = ""
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        if pending:
            combined = f"{pending}\n\n{text}".strip()
            if len(combined) <= max_chars:
                pending = combined
                continue
            merged.append(pending)
            pending = text
            continue
        if len(text) < merge_threshold and merged:
            combined = f"{merged[-1]}\n\n{text}".strip()
            if len(combined) <= max_chars:
                merged[-1] = combined
                continue
        pending = text
    if pending:
        if merged and len(pending) < merge_threshold:
            combined = f"{merged[-1]}\n\n{pending}".strip()
            if len(combined) <= max_chars:
                merged[-1] = combined
            else:
                merged.append(pending)
        else:
            merged.append(pending)
    return merged


def chunk_section(
    section: DocumentSection,
    *,
    source_section_index: int,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
    min_section_chars: int = DEFAULT_MIN_SECTION_CHARS,
    fallback_overlap: int = 0,
) -> list[SectionChunk]:
    body = _section_body(section)
    if not body:
        return []

    if len(body) <= max_section_chars:
        return [
            SectionChunk(
                heading=section.heading,
                level=section.level,
                text=body,
                source_section_index=source_section_index,
                chunk_index=0,
            )
        ]

    split_chunks = _split_paragraphs(
        body,
        max_chars=max_section_chars,
        fallback_overlap=fallback_overlap,
    )
    split_chunks = _merge_small_chunks(
        split_chunks,
        max_chars=max_section_chars,
        merge_threshold=min(min_section_chars, DEFAULT_MERGE_THRESHOLD),
    )
    return [
        SectionChunk(
            heading=section.heading,
            level=section.level,
            text=text,
            source_section_index=source_section_index,
            chunk_index=chunk_index,
        )
        for chunk_index, text in enumerate(split_chunks)
        if text.strip()
    ]


def chunk_document(
    document: Document,
    *,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
    min_section_chars: int = DEFAULT_MIN_SECTION_CHARS,
    fallback_overlap: int = 0,
) -> list[SectionChunk]:
    """Split a Document into heading-aware chunks sized for ranking and prompts."""
    chunks: list[SectionChunk] = []
    for section_index, section in enumerate(document.sections):
        chunks.extend(
            chunk_section(
                section,
                source_section_index=section_index,
                max_section_chars=max_section_chars,
                min_section_chars=min_section_chars,
                fallback_overlap=fallback_overlap,
            )
        )
    return chunks
