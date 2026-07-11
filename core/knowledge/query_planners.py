"""Query planner templates for generic orchestration pipeline."""

from __future__ import annotations

import re
from typing import Any

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
    }
)


def plan_query(
    query: str,
    *,
    semantic_query: str | None = None,
    planner: str = "passthrough",
) -> dict[str, Any]:
    raw = (query or "").strip()
    semantic = (semantic_query or raw).strip()
    planner_id = (planner or "passthrough").strip().lower()

    if planner_id == "keyword_extract":
        tokens = [
            t
            for t in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", semantic.lower())
            if t not in _STOPWORDS
        ]
        search_query = " ".join(tokens[:12]) or semantic
    elif planner_id == "entity_centric":
        caps = re.findall(r"\b[A-Z][a-zA-Z0-9._-]{1,}\b", raw)
        search_query = " ".join(caps[:6]) or semantic
    else:
        search_query = semantic

    return {
        "query_raw": raw,
        "semantic_query": semantic,
        "search_query": search_query,
        "planner": planner_id,
    }
