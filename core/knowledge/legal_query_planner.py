"""Deterministic query planning for legal_knowledge (@legal / CourtListener)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.adapters.query_sanitize import sanitize_api_query

_CONVERSATIONAL_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^what\s+did\s+(?:the\s+)?(?:u\.?\s*s\.?\s*)?supreme\s+court\s+hold\s+(?:in|about)\s+",
        re.IGNORECASE,
    ),
    re.compile(r"^what\s+(?:are|is|was|were)\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^show\s+me\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^find\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^list\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^summarize\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^describe\s+", re.IGNORECASE),
    re.compile(r"^explain\s+(?:the\s+)?", re.IGNORECASE),
)

_CASE_NAME = re.compile(
    r"\b([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3})\s+v\.?\s+"
    r"([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3})\b",
)

_CASE_PARTY_STOPWORDS = frozenset(
    {
        "about",
        "during",
        "hold",
        "in",
        "the",
        "supreme",
        "court",
        "what",
        "did",
        "for",
        "from",
        "with",
        "and",
        "or",
    }
)

_TRAILING_FILLER = re.compile(
    r"\s+(?:please|thanks|thank\s+you)\s*$",
    re.IGNORECASE,
)

_LEGAL_NOISE = re.compile(
    r"\b(?:court\s*listener|case\s*law|legal\s+precedent|precedent|"
    r"opinion|court\s+opinion|ruling|decision|judgment|judgement)\b",
    re.IGNORECASE,
)

_SCOTUS_HINT = re.compile(
    r"\b(?:supreme\s+court|scotus|u\.?\s*s\.?\s*supreme\s+court)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegalQueryPlan:
    search_query: str
    semantic_query: str


def _normalize_party(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def extract_case_name(text: str) -> str | None:
    """Return a normalized 'Plaintiff v. Defendant' case caption when present."""
    candidates: list[tuple[str, str]] = []
    for match in _CASE_NAME.finditer(text or ""):
        left = match.group(1).strip()
        right = match.group(2).strip()
        if not left or not right:
            continue
        left_words = {word.lower() for word in left.split()}
        right_words = {word.lower() for word in right.split()}
        if left_words & _CASE_PARTY_STOPWORDS:
            continue
        if right_words & _CASE_PARTY_STOPWORDS:
            continue
        candidates.append((left, right))
    if not candidates:
        return None
    left, right = candidates[-1]
    return f"{left} v. {right}"


def extract_case_name_key(text: str) -> tuple[str, str] | None:
    """Normalized party pair for ranking CourtListener search hits."""
    caption = extract_case_name(text)
    if not caption:
        return None
    match = _CASE_NAME.search(caption)
    if not match:
        return None
    return (_normalize_party(match.group(1)), _normalize_party(match.group(2)))


def _strip_conversational_phrasing(text: str) -> str:
    q = _TRAILING_FILLER.sub("", (text or "").strip())
    for pattern in _CONVERSATIONAL_PREFIXES:
        q = pattern.sub("", q).strip()
    q = _LEGAL_NOISE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return sanitize_api_query(q)


def _append_court_filter(search: str, raw: str) -> str:
    if _SCOTUS_HINT.search(raw) and "court_id:" not in search.lower():
        return f"{search} court_id:scotus".strip()
    return search


def plan_legal_query(
    query: str,
    *,
    semantic_query: str | None = None,
) -> LegalQueryPlan:
    """Map conversational @legal prompts to CourtListener search terms."""
    raw = sanitize_api_query(query)
    semantic = sanitize_api_query(semantic_query or query)
    case_name = extract_case_name(raw)
    if case_name:
        search = case_name
    else:
        stripped = _strip_conversational_phrasing(raw)
        search = stripped or semantic or raw
    search = _append_court_filter(search, raw)
    return LegalQueryPlan(
        search_query=search,
        semantic_query=semantic,
    )
