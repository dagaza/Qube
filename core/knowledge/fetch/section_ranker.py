"""Rank section chunks and map Documents to EvidenceObject lists."""

from __future__ import annotations

import difflib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.knowledge.document.types import Document
from core.knowledge.fetch.section_chunker import (
    DEFAULT_MAX_SECTION_CHARS,
    SectionChunk,
    chunk_document,
)
from core.knowledge.ranking.authority import authority_score_for_url
from core.knowledge.ranking.relevance import score_evidence_row
from core.knowledge.types import EvidenceObject

MMR_LAMBDA = 0.72
DEFAULT_MAX_RESULTS = 3
EXCERPT_MAX_CHARS = 800
FULL_TEXT_MAX_CHARS = 4000


@dataclass(frozen=True)
class RankedSectionChunk:
    chunk: SectionChunk
    relevance_score: float


def _chunk_row(chunk: SectionChunk, *, document: Document) -> dict[str, Any]:
    title_bits = [document.title or "", chunk.heading or ""]
    title = " — ".join(bit.strip() for bit in title_bits if bit and bit.strip())
    return {
        "title": title or document.url,
        "snippet": chunk.text,
        "full_text": chunk.text,
        "url": document.url,
        "_section_heading": chunk.heading,
        "_section_level": chunk.level,
        "_source_section_index": chunk.source_section_index,
        "_chunk_index": chunk.chunk_index,
    }


def _chunk_similarity(a: SectionChunk, b: SectionChunk) -> float:
    left = f"{a.heading or ''}\n{a.text}".strip().lower()
    right = f"{b.heading or ''}\n{b.text}".strip().lower()
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def mmr_select_chunks(
    ranked: list[RankedSectionChunk],
    *,
    max_results: int,
    lambda_: float = MMR_LAMBDA,
) -> list[RankedSectionChunk]:
    """Maximal Marginal Relevance over ranked section chunks."""
    if not ranked or max_results <= 0:
        return []
    if len(ranked) <= max_results:
        return list(ranked)

    rel_scores = [item.relevance_score for item in ranked]
    lo = min(rel_scores)
    hi = max(rel_scores)
    span = hi - lo

    def _norm_rel(score: float) -> float:
        if span <= 1e-9:
            return 1.0 if score > 0 else 0.0
        return (score - lo) / span

    selected: list[RankedSectionChunk] = []
    remaining = list(ranked)

    while remaining and len(selected) < max_results:
        best_idx = 0
        best_mmr = float("-inf")
        for idx, candidate in enumerate(remaining):
            rel = _norm_rel(candidate.relevance_score)
            max_sim = 0.0
            for picked in selected:
                max_sim = max(max_sim, _chunk_similarity(candidate.chunk, picked.chunk))
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def rank_section_chunks(
    chunks: list[SectionChunk],
    *,
    document: Document,
    query: str,
    semantic_query: str | None = None,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_score: float = 0.05,
) -> list[RankedSectionChunk]:
    """Score and MMR-rank section chunks against the turn query."""
    if not chunks:
        return []

    scored: list[RankedSectionChunk] = []
    effective_query = (semantic_query or query or "").strip() or query
    for chunk in chunks:
        row = _chunk_row(chunk, document=document)
        relevance = score_evidence_row(
            row,
            query=effective_query,
            query_vector=query_vector,
            embed_fn=embed_fn,
        )
        if relevance < min_score:
            continue
        scored.append(RankedSectionChunk(chunk=chunk, relevance_score=relevance))

    scored.sort(key=lambda item: item.relevance_score, reverse=True)
    return mmr_select_chunks(scored, max_results=max_results)


def _section_title(document: Document, chunk: SectionChunk) -> str:
    if chunk.heading and document.title:
        return f"{document.title} — {chunk.heading}"
    if chunk.heading:
        return chunk.heading
    return document.title or document.url


def _section_to_evidence(
    document: Document,
    ranked: RankedSectionChunk,
    *,
    index: int,
    retrieved_at: float,
    adapter: str = "fetch_engine",
    retrieval_method: str = "fetch_extract",
    document_type: str = "web_section",
) -> EvidenceObject:
    chunk = ranked.chunk
    section_text = chunk.text.strip()
    excerpt = section_text[:EXCERPT_MAX_CHARS]
    full_text = section_text if len(section_text) <= FULL_TEXT_MAX_CHARS else None
    metadata = document.metadata
    structured_data = document.structured_data if chunk.source_section_index == 0 else {}
    raw_metadata: dict[str, Any] = {
        "section_index": chunk.source_section_index,
        "chunk_index": chunk.chunk_index,
        "section_heading": chunk.heading,
        "section_level": chunk.level,
        "structured_data": structured_data or None,
    }
    if metadata is not None:
        raw_metadata.update(
            {
                "extractor_name": metadata.extractor_name,
                "extractor_version": metadata.extractor_version,
                "extractor_confidence": metadata.extractor_confidence,
                "fetch_tier": metadata.fetch_tier,
            }
        )

    authority = authority_score_for_url(document.url)
    source_id = f"{document.url}#section-{chunk.source_section_index}-{chunk.chunk_index}"
    return EvidenceObject(
        id=f"ws_{index}",
        source_id=source_id,
        adapter=adapter,
        retrieval_method=retrieval_method,
        title=_section_title(document, chunk),
        excerpt=excerpt,
        full_text=full_text,
        url=document.url,
        document_type=document_type,
        relevance_score=max(0.0, min(1.0, ranked.relevance_score)),
        authority_score=authority,
        reliability_score=max(0.0, min(1.0, ranked.relevance_score * 0.85)),
        retrieved_at=retrieved_at,
        fetch_status="full_extract",
        raw_metadata=raw_metadata,
    )


def document_to_evidence_objects(
    document: Document,
    *,
    query: str,
    semantic_query: str | None = None,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
    adapter: str = "fetch_engine",
    retrieval_method: str = "fetch_extract",
    document_type: str = "web_section",
) -> list[EvidenceObject]:
    """Chunk, rank, and map a Document to EvidenceObject sections."""
    chunks = chunk_document(document, max_section_chars=max_section_chars)
    ranked = rank_section_chunks(
        chunks,
        document=document,
        query=query,
        semantic_query=semantic_query,
        query_vector=query_vector,
        embed_fn=embed_fn,
        max_results=max_results,
    )
    retrieved_at = time.time()
    return [
        _section_to_evidence(
            document,
            item,
            index=index,
            retrieved_at=retrieved_at,
            adapter=adapter,
            retrieval_method=retrieval_method,
            document_type=document_type,
        )
        for index, item in enumerate(ranked, start=1)
    ]


def build_section_evidence_bundle_id() -> str:
    return f"bundle_{uuid.uuid4().hex[:12]}"
