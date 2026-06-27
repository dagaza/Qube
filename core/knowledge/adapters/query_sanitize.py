"""Normalize user queries before external API adapter calls."""

from __future__ import annotations

import re

_MAX_LEN = 500

_TRAILING_PUNCT_RE = re.compile(r"[\s?!.;,:]+$")
_MULTI_SPACE_RE = re.compile(r"\s+")


def sanitize_api_query(query: str, *, max_len: int = _MAX_LEN) -> str:
    """Strip trailing punctuation and collapse whitespace for API search params."""
    q = (query or "").strip().strip("\"'")
    q = _TRAILING_PUNCT_RE.sub("", q).strip()
    q = _MULTI_SPACE_RE.sub(" ", q)
    if max_len > 0:
        q = q[:max_len].strip()
    return q
