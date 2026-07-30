"""Pro precision ingest — embedding-similarity breakpoint chunking."""

from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np

from core.chunking.ingest_pipeline import chunk_document_for_ingest
from core.chunking.structure_chunker import ChunkRecord
from core.knowledge.document.types import Document
from rag.chunker import chunk_text

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_SIMILARITY_THRESHOLD = 0.45
_MIN_SEMANTIC_BODY_CHARS = 400


class _Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def split_sentences(text: str) -> list[str]:
    body = (text or "").strip()
    if not body:
        return []
    parts = _SENTENCE_SPLIT_RE.split(body)
    return [part.strip() for part in parts if part.strip()]


def semantic_breakpoint_chunks(
    text: str,
    embedder: _Embedder,
    *,
    max_chars: int = 1500,
    min_chars: int = 120,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    overlap: int = 150,
) -> list[str]:
    """
    Split ``text`` at embedding-similarity valleys between consecutive sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        body = sentences[0]
        if len(body) <= max_chars:
            return [body]
        return chunk_text(body, chunk_size=max_chars, overlap=overlap)

    vectors = np.asarray(embedder.embed(sentences), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
        return chunk_text(text, chunk_size=max_chars, overlap=overlap)

    breakpoints = [0]
    for index in range(len(sentences) - 1):
        similarity = float(np.dot(vectors[index], vectors[index + 1]))
        if similarity < similarity_threshold:
            breakpoints.append(index + 1)
    breakpoints.append(len(sentences))

    chunks: list[str] = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        segment = " ".join(sentences[start:end]).strip()
        if not segment:
            continue
        if len(segment) > max_chars:
            chunks.extend(chunk_text(segment, chunk_size=max_chars, overlap=overlap))
            continue
        if len(segment) < min_chars and chunks:
            chunks[-1] = f"{chunks[-1]} {segment}".strip()
        else:
            chunks.append(segment)
    return [chunk for chunk in chunks if chunk.strip()]


def _remap_records_with_bodies(
    template: ChunkRecord,
    bodies: list[str],
) -> list[ChunkRecord]:
    if not bodies:
        return []
    if len(bodies) == 1:
        return [template]
    return [
        ChunkRecord(
            body=body,
            heading=template.heading,
            heading_level=template.heading_level,
            breadcrumb=template.breadcrumb,
            section_index=template.section_index,
            chunk_index=index,
            page_start=template.page_start,
            page_end=template.page_end,
        )
        for index, body in enumerate(bodies)
    ]


def chunk_document_for_precision_ingest(
    document: Document,
    embedder: Any,
    **ingest_kwargs,
) -> list[ChunkRecord]:
    """
    Structural chunking followed by semantic re-segmentation of large bodies.
    """
    base_records = chunk_document_for_ingest(document, **ingest_kwargs)
    if not base_records:
        return base_records

    expanded: list[ChunkRecord] = []
    for record in base_records:
        body = (record.body or "").strip()
        if len(body) < _MIN_SEMANTIC_BODY_CHARS:
            expanded.append(record)
            continue
        bodies = semantic_breakpoint_chunks(body, embedder)
        expanded.extend(_remap_records_with_bodies(record, bodies))

    if not expanded:
        return expanded

    total = len(expanded)
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
        for record in expanded
    ]
