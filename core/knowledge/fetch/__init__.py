"""HTTP fetch subsystem — transport layer before extraction."""

from core.knowledge.fetch.blockers import detect_blocker
from core.knowledge.fetch.engine import fetch_html_string, fetch_url
from core.knowledge.fetch.section_chunker import (
    DEFAULT_MAX_SECTION_CHARS,
    SectionChunk,
    chunk_document,
    chunk_section,
)
from core.knowledge.fetch.section_ranker import (
    RankedSectionChunk,
    document_to_evidence_objects,
    mmr_select_chunks,
    rank_section_chunks,
)
from core.knowledge.fetch.types import BlockerReason, FetchResult

__all__ = [
    "BlockerReason",
    "DEFAULT_MAX_SECTION_CHARS",
    "FetchResult",
    "RankedSectionChunk",
    "SectionChunk",
    "chunk_document",
    "chunk_section",
    "detect_blocker",
    "document_to_evidence_objects",
    "fetch_html_string",
    "fetch_url",
    "mmr_select_chunks",
    "rank_section_chunks",
]
