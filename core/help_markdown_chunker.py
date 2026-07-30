"""Heading-aware chunking for help corpus markdown (§3.8)."""

from __future__ import annotations

from core.chunking.markdown_sections import split_markdown_sections
from core.chunking.library_chunking import chunk_markdown_text

_DEFAULT_MAX_CHARS = 1500

# Back-compat alias for existing imports/tests.
split_help_markdown_sections = split_markdown_sections


def chunk_help_markdown(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[str]:
    """
    Chunk help docs at semantic heading boundaries.

    Long sections are further split with the standard RAG chunker hard cap.
    """
    return chunk_markdown_text(text, max_chars=max_chars)
