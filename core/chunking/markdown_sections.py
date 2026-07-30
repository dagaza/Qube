"""Markdown heading boundaries shared by Library ingest and help corpus."""

from __future__ import annotations

import re

# Help corpus: H2/H3 only (legacy behavior).
_HEADING_SPLIT_H2_H3_RE = re.compile(r"(?=^#{2,3}\s+)", re.MULTILINE)

# Library ingest: all ATX heading levels.
_HEADING_SPLIT_ALL_RE = re.compile(r"(?=^#{1,6}\s+)", re.MULTILINE)


def _split_on_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    body = (text or "").strip()
    if not body:
        return []
    parts = [part.strip() for part in pattern.split(body) if part.strip()]
    if not parts:
        return [body]
    if parts[0].startswith("# ") and not parts[0].startswith("##"):
        # Drop lone H1 title block when followed by sections.
        if len(parts) > 1:
            parts = parts[1:]
    return parts


def split_markdown_sections(text: str) -> list[str]:
    """Split markdown on H2/H3 boundaries (help corpus default)."""
    return _split_on_pattern(text, _HEADING_SPLIT_H2_H3_RE)


def split_markdown_sections_all_levels(text: str) -> list[str]:
    """Split markdown on H1–H6 boundaries (Library ingest)."""
    return _split_on_pattern(text, _HEADING_SPLIT_ALL_RE)
