"""Deterministic query planning for finance_knowledge (@finance / SEC EDGAR)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.adapters.query_sanitize import sanitize_api_query

_CONVERSATIONAL_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^what\s+(?:are|is)\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^show\s+me\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^find\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^list\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^summarize\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^describe\s+", re.IGNORECASE),
)

_FORM_TYPE_RE = re.compile(
    r"\b(10-K/A|10-K|10-Q/A|10-Q|8-K|20-F|6-K|S-1|DEF\s*14A)\b",
    re.IGNORECASE,
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

_TRAILING_FILLER = re.compile(
    r"\s+(?:please|thanks|thank\s+you)\s*$",
    re.IGNORECASE,
)

_SEC_NOISE = re.compile(
    r"\b(?:sec|edgar|filings?|filing|annual\s+report|quarterly\s+report)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FinanceQueryPlan:
    search_query: str
    semantic_query: str
    form_types: tuple[str, ...] = ()


def _strip_conversational_phrasing(text: str) -> str:
    q = _TRAILING_FILLER.sub("", (text or "").strip())
    for pattern in _CONVERSATIONAL_PREFIXES:
        q = pattern.sub("", q).strip()
    q = _SEC_NOISE.sub(" ", q)
    q = _FORM_TYPE_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return sanitize_api_query(q)


def _extract_form_types(text: str) -> tuple[str, ...]:
    forms = [m.group(1).upper().replace(" ", "") for m in _FORM_TYPE_RE.finditer(text or "")]
    return tuple(dict.fromkeys(forms))


def plan_finance_query(
    query: str,
    *,
    semantic_query: str | None = None,
) -> FinanceQueryPlan:
    """Map conversational @finance prompts to SEC search terms."""
    raw = sanitize_api_query(query)
    semantic = sanitize_api_query(semantic_query or query)
    form_types = _extract_form_types(raw)
    stripped = _strip_conversational_phrasing(raw)
    search = stripped or semantic or raw
    return FinanceQueryPlan(
        search_query=search,
        semantic_query=semantic,
        form_types=form_types,
    )
