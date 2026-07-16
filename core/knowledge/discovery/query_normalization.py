"""Canonical discovery query strings for cache keys and dedup."""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"\s+")


def normalize_discovery_query(query: str) -> str:
    """Normalize a query for discovery cache keys.

    Lowercase, collapse whitespace, strip trailing question marks.
    The original query is still sent to providers; this form is for
    cache/dedup identity only.
    """
    text = (query or "").strip().lower()
    text = _MULTI_SPACE.sub(" ", text)
    while text.endswith("?"):
        text = text[:-1].strip()
    return text
