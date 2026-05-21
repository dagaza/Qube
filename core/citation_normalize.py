"""Normalize model citation tokens before Qt markdown linkification."""
from __future__ import annotations

import re

_W_LABELED_CITE = re.compile(r"\[W:\s*[^\]]+\]", re.IGNORECASE)
_NUM_LABELED_CITE = re.compile(r"\[(\d+):\s*[^\]]+\]")


def normalize_labeled_citation_tokens(text: str) -> str:
    """
    Map echoed SOURCE headers like ``[W: Live Web Search]`` → ``[W]`` so UI
    citation linkifiers can match stored source ids.
    """
    if not text:
        return text
    t = _W_LABELED_CITE.sub("[W]", text)
    return _NUM_LABELED_CITE.sub(r"[\1]", t)
