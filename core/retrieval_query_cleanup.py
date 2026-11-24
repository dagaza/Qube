"""Strip conversational request frames from retrieval queries (plan S5).

"Yes, please tell me how ANT analyzes AI for example." searches better as
"how ANT analyzes AI". This removes only *frames* around the question, and only
at phrase boundaries:

* leading affirmations followed by punctuation ("Yes,", "Sure.", "OK!");
* leading "please", "tell me [more] [about]" before a question word or "about",
  and "can/could/would you [please] explain/tell me/describe/…";
* a trailing "for example" / "for instance" / "please" / "thanks";
* stray markup-like punctuation such as "?>" left by a slip of the keyboard.

Nothing inside the question is touched ("example-based learning", "Tell Me
Lies" stay), and when stripping would leave no words the original text is
returned unchanged. Used by the scientific planner (keyword *and* semantic
queries) and the general-web / trusted discovery queries. Not user-visible.

``recency_need`` (plan S7, R9) flags questions that ask for current
information ("latest", "current", "today", a recent year, a price / rate /
score) and returns the freshness window sources must fall in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

_AFFIRMATION = re.compile(
    r"^(?:yes|yeah|yep|sure|ok|okay|alright|great|thanks|thank\s+you)\s*[,.!;:]+\s*",
    re.IGNORECASE,
)
# "please, …" / "please tell me …" — not "Please Please Me".
_PLEASE = re.compile(
    r"^please\s*(?:,\s*|(?=(?:tell|explain|describe|summari[sz]e|show|give|find|search|look|list|"
    r"what|how|why|when|where|who|which|is|are|do|does|can|could)\b))",
    re.IGNORECASE,
)
_CAN_YOU = re.compile(
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:explain|describe|summari[sz]e|tell\s+me|show\s+me|give\s+me)\b"
    r"(?:\s+(?:more\s+)?(?:about|on)\b)?[\s,]*",
    re.IGNORECASE,
)
# "tell me about X" / "tell me how X …" — not "Tell Me Lies".
_TELL_ME = re.compile(
    r"^tell\s+me\s+(?:more\s+)?(?:about\s+|(?=(?:how|what|why|when|where|who|which|whether|if)\b))",
    re.IGNORECASE,
)
_LEADING = (_AFFIRMATION, _PLEASE, _CAN_YOU, _TELL_ME)

_TRAILING = re.compile(
    r"(?:[\s,;]+(?:for\s+example|for\s+instance|please|thanks|thank\s+you))+\s*([?!.]*)\s*$",
    re.IGNORECASE,
)
# "theory?>" / "theory>" at the end, or "?>" right after punctuation.
_STRAY_MARKUP = re.compile(r"(?:(?<=[?!.])\s*[<>]+|\s*[<>]+\s*$)")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")
# What is left after stripping must be more than a frame ("Yes, please." stays).
_ONLY_FRAME = re.compile(r"^(?:please|thanks|thank\s+you|ok|okay|sure|yes)?[\s.!?,]*$", re.IGNORECASE)


def strip_request_frames(text: str) -> str:
    """Return ``text`` without leading/trailing request frames (see module doc)."""
    original = (text or "").strip()
    if not original:
        return original
    q = _STRAY_MARKUP.sub("", original).strip()
    changed = True
    while changed:
        changed = False
        for pattern in _LEADING:
            stripped = pattern.sub("", q, count=1)
            if stripped != q:
                q = stripped.strip()
                changed = True
    q = _TRAILING.sub(lambda m: m.group(1), q).strip()
    q = re.sub(r"^[\s,;:]+", "", q)
    if q == original:
        return original
    # A statement's final period is noise for search; "?" is kept.
    q = re.sub(r"\s*\.+$", "", q).strip()
    if not _HAS_WORD.search(q) or _ONLY_FRAME.match(q):
        return original
    return q


# ---------------------------------------------------------------------------
# Recency need (plan S7, R9)
# ---------------------------------------------------------------------------

RECENCY_WINDOW_DAYS = 365
NEWS_WINDOW_DAYS = 7

# News-like phrasing: the answer changes within days.
_NEWS = re.compile(
    r"\b(?:today|tonight|yesterday|this\s+(?:morning|afternoon|evening|week)|breaking|"
    r"right\s+now|latest\s+news|news\s+(?:on|about|from|today)|headlines?|"
    r"live\s+(?:scores?|updates?|coverage))\b",
    re.IGNORECASE,
)
_RECENT_WORDS = re.compile(
    r"\b(?:latest|newest|most\s+recent|recent\s+(?:news|developments?|changes?|updates?)|"
    r"recently|up[-\s]to[-\s]date|nowadays|these\s+days|at\s+the\s+moment|"
    r"this\s+(?:year|month)|so\s+far\s+this|as\s+of\s+(?:now|today))\b",
    re.IGNORECASE,
)
# "current(ly)" except the physics / ocean noun ("electric current", "current density").
_CURRENT = re.compile(r"\bcurrent(?:ly)?\b", re.IGNORECASE)
_CURRENT_NOUN_BEFORE = re.compile(
    r"\b(?:electric(?:al)?|alternating|direct|eddy|ocean|rip|tidal|leakage|displacement|"
    r"induced|ac|dc|of|the\s+ocean|a)\s+$",
    re.IGNORECASE,
)
_CURRENT_NOUN_AFTER = re.compile(
    r"^\s+(?:density|flow|source|loop|limit(?:ing)?|sensor|divider|transformer|carrying|"
    r"clamp|draw|gain|mirror|amplifier|through|flow(?:s|ing)?|in\s+a\s+(?:wire|circuit))\b",
    re.IGNORECASE,
)
# Price / rate / score questions whose answer moves over months.
_MARKET = re.compile(
    r"\b(?:price\s+(?:of|for)|prices\s+(?:of|for)|stock\s+price|share\s+price|"
    r"(?:exchange|interest|inflation|mortgage|deposit(?:\s+facility)?|unemployment|"
    r"policy|base|bank|prime|savings|cpi)\s+rates?|"
    r"(?:final|latest|live)\s+score|who\s+(?:won|is\s+winning))\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<![\d.])((?:19|20)\d{2})(?![\d.])")


@dataclass(frozen=True)
class RecencyNeed:
    """The question asks for current information (plan S7, R9)."""

    window_days: int
    reason: str  # "news" | "recent_wording" | "current" | "market" | "recent_year"


def _current_is_temporal(text: str) -> bool:
    for m in _CURRENT.finditer(text):
        if m.group(0).lower() == "currently":
            return True
        if _CURRENT_NOUN_BEFORE.search(text[: m.start()]):
            continue
        if _CURRENT_NOUN_AFTER.match(text[m.end():]):
            continue
        return True
    return False


def recency_need(text: str, *, today: date | None = None) -> RecencyNeed | None:
    """Whether ``text`` asks for current information, and the window sources need.

    Deterministic and English-only (D4: non-English questions keep the old
    behaviour). News-like phrasing ("today", "this week", "breaking") gets a
    7-day window; "latest / current / newest", price / rate / score questions
    and a mention of this or last year get 12 months. ``None`` otherwise —
    including "electric current", "price elasticity" and historical years.
    """
    q = " ".join((text or "").split())
    if not q:
        return None
    if _NEWS.search(q):
        return RecencyNeed(NEWS_WINDOW_DAYS, "news")
    if _RECENT_WORDS.search(q):
        return RecencyNeed(RECENCY_WINDOW_DAYS, "recent_wording")
    if _current_is_temporal(q):
        return RecencyNeed(RECENCY_WINDOW_DAYS, "current")
    year_now = (today or datetime.now(UTC).date()).year
    years = [int(m.group(1)) for m in _YEAR.finditer(q)]
    # "What was the inflation rate in 2019?", "Who won the 2022 World Cup?":
    # a historical year pins the question to the past.
    historical = any(y < year_now - 1 for y in years)
    if _MARKET.search(q) and not historical:
        return RecencyNeed(RECENCY_WINDOW_DAYS, "market")
    if any(year_now - 1 <= y <= year_now for y in years):
        return RecencyNeed(RECENCY_WINDOW_DAYS, "recent_year")
    return None


__all__ = [
    "NEWS_WINDOW_DAYS",
    "RECENCY_WINDOW_DAYS",
    "RecencyNeed",
    "recency_need",
    "strip_request_frames",
]
