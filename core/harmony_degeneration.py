"""
Detect gpt-oss / Harmony completion degeneration (untagged planning loops, broken markdown).

Shared by the stream parser, artifact stripper, and native worker end-of-turn sanitize.
"""
from __future__ import annotations

import re
from typing import Optional

# Log-derived planning / loop tails (never valid user-facing answer text).
_PLANNING_TAIL = re.compile(
    r"(?is)"
    r"(?:"
    r"\n\s*\*\*\s*we\s*(?:…|\.{2,})"
    r"|"
    r"\n\s*we\s*(?:…|\.{2,}|[.\?\‐‑–—\s]){3,}"
    r"|"
    r"\n\s*the\s*(?:…|\.{2,}|[.\?\‐‑–—\s]){2,}"
    r"|"
    r"\n\s*this\s*(?:…|\.{2,})"
    r"|"
    r"\n\s*in‑‑"
    r"|"
    r"\n\s*we\s*\n"
    r"|"
    r"(?<=[.!?…]\s)we\s*(?:…|\.{2,}|\s*[.\?\‐‑–—]){4,}"
    r")"
)

# Broken numbered markdown line (gpt-oss planning), not a normal second bullet.
_BROKEN_NUMBERED_LINE = re.compile(
    r"(?is)"
    r"\n\s*\d+\.\s*\*\*[^\n]*(?:\?\?|\*\*\s*‑\s*\d+\s*–\s*–|‑\s*\d+\s*–\s*–)"
    r"|"
    r"\n\s*\d+\.\s*\*\*[^\n]{0,72}?‑‑"
    r"|"
    r"\*\*Pre‑‑treatment"
)

# Trailing clause fragments when generation was cut mid-sentence.
_DANGLING_CLAUSE_TAIL = re.compile(
    r"(?is)\s*,?\s*that often\b.*$|[,;]\s*especially\b[^.!?]{0,120}$"
)


def find_degeneration_start(text: str) -> Optional[int]:
    """Return the index where degeneration begins, or None if the text looks clean."""
    if not text:
        return None
    earliest: Optional[int] = None
    for pat in (_BROKEN_NUMBERED_LINE, _PLANNING_TAIL):
        m = pat.search(text)
        if m is None:
            continue
        if earliest is None or m.start() < earliest:
            earliest = m.start()
    return earliest


def polish_harmony_visible_text(text: str) -> str:
    """Truncate at degeneration, then drop common gpt-oss dangling clause tails."""
    t = truncate_at_degeneration(text)
    if not t:
        return t
    t = _DANGLING_CLAUSE_TAIL.sub("", t)
    return t.rstrip()


def truncate_at_degeneration(text: str) -> str:
    """Keep only the user-visible prefix before the first degeneration marker."""
    cut = find_degeneration_start(text)
    if cut is None:
        return text
    return text[:cut].rstrip()


_PUNCT_RUN = re.compile(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{24,}")
_PUNCT_CHARS = re.compile(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]")


_ORPHAN_LIST_FRAGMENT = re.compile(
    r"^\s*(?:\n+)?\d{1,2}\.?\s*$"
)


def is_harmony_orphan_stream_fragment(fragment: str) -> bool:
    """
    Drop streamed atoms like ``2`` or ``\\n\\n2.`` that precede a degeneration cut.

    gpt-oss often emits the list index to TTS before the ``**`` garbage is visible.
    """
    if not fragment or not fragment.strip():
        return False
    return bool(_ORPHAN_LIST_FRAGMENT.match(fragment))


def harmony_tail_degenerate(text: str, *, tail_chars: int = 500) -> bool:
    """True when the recent completion tail is a gpt-oss punctuation / We loop."""
    if not text:
        return False
    if find_degeneration_start(text) is not None:
        return True
    tail = text[-tail_chars:]
    if _PUNCT_RUN.search(tail):
        return True
    spaced = re.search(
        r"[\s\u00a0\u202f\u2009\u200a\u2028.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{28,}",
        tail,
    )
    if spaced is not None and len(_PUNCT_CHARS.findall(spaced.group(0))) >= 12:
        return True
    return False
