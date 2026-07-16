"""Heading-aware section chunking for fetched Documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.document.types import Document, DocumentSection

DEFAULT_MAX_SECTION_CHARS = 800
DEFAULT_MIN_SECTION_CHARS = 200
DEFAULT_MERGE_THRESHOLD = 150

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SectionChunk:
    heading: str | None
    level: int
    text: str
    source_section_index: int
    chunk_index: int = 0


def _section_body(section: DocumentSection) -> str:
    parts: list[str] = []
    if section.text:
        parts.append(section.text.strip())
    if section.list_items:
        parts.extend(f"- {item.strip()}" for item in section.list_items if item.strip())
    return "\n".join(parts).strip()


def _split_sentences(text: str, *, max_chars: int) -> list[str]:
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


def _split_paragraphs(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return _split_sentences(text, max_chars=max_chars)

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_sentences(paragraph, max_chars=max_chars))
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

    split_chunks = _split_paragraphs(body, max_chars=max_section_chars)
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
            )
        )
    return chunks
