"""Heuristic query normalization and multi-angle sub-query expansion for deep research."""

from __future__ import annotations

import re
from typing import Callable

MAX_SUB_QUERIES = 3

_QUERY_CORRECTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmace inhibitors?\b", re.I), "ACE inhibitors"),
    (re.compile(r"\bace inhib\b", re.I), "ACE inhibitor"),
)

_CLINICAL_SIGNALS = frozenset(
    {
        "heart failure",
        "hf",
        "cardiovascular",
        "mortality",
        "survival",
        "outcomes",
        "trial",
        "evidence",
        "efficacy",
        "gdmt",
        "ace inhibitor",
        "ace inhibitors",
        "arb",
        "beta blocker",
        "semaglutide",
        "sglt2",
    }
)


def normalize_deep_research_query(query: str) -> str:
    """Fix common typos and normalize whitespace before decomposition."""
    text = re.sub(r"\s+", " ", (query or "").strip())
    for pattern, replacement in _QUERY_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _has_clinical_signal(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in _CLINICAL_SIGNALS)


def expand_research_angles(query: str, *, max_angles: int = MAX_SUB_QUERIES) -> tuple[str, ...]:
    """Add bounded retrieval angles for single short clinical queries."""
    base = normalize_deep_research_query(query)
    if not base:
        return ()

    lower = base.lower()
    angles: list[str] = [base]

    if _has_clinical_signal(base):
        if "randomized" not in lower and "rct" not in lower:
            angles.append(f"{base} randomized controlled trial")
        if "ace inhibitor" in lower:
            angles.append(
                "ACE inhibitors enalapril ramipril lisinopril heart failure mortality trial"
            )
        elif "meta-analysis" not in lower and "systematic review" not in lower:
            angles.append(f"{base} systematic review meta-analysis")
        if "mortality" not in lower and "heart failure" in lower:
            angles.append(f"{base} mortality hospitalization outcomes")
    elif "evidence" in lower:
        angles.append(re.sub(r"\bevidence\b", "clinical trial outcomes", base, flags=re.I))

    deduped = list(dict.fromkeys(a.strip() for a in angles if len(a.strip()) >= 12))
    if not deduped:
        deduped = [base]
    return tuple(deduped[: max(1, max_angles)])


def decompose_query(
    query: str,
    *,
    max_sub_queries: int = MAX_SUB_QUERIES,
    generate_fn: Callable[[str, str], str] | None = None,
) -> tuple[str, ...]:
    """Bounded sub-query split; optional LLM planner when ``generate_fn`` is provided."""
    if generate_fn is not None:
        from core.knowledge.deep_research_decompose_llm import decompose_query_with_llm

        return decompose_query_with_llm(
            query,
            generate_fn,
            max_sub_queries=max_sub_queries,
        )

    return _decompose_query_heuristic(query, max_sub_queries=max_sub_queries)


def _decompose_query_heuristic(
    query: str,
    *,
    max_sub_queries: int = MAX_SUB_QUERIES,
) -> tuple[str, ...]:
    """Heuristic split for multi-angle retrieval."""
    raw = normalize_deep_research_query(query)
    if not raw:
        return ()

    parts: list[str] = []
    for chunk in raw.replace(";", ".").split("."):
        chunk = chunk.strip()
        if not chunk:
            continue
        for piece in chunk.split(" and "):
            piece = piece.strip(" ,")
            if len(piece) >= 12:
                parts.append(piece)

    if not parts:
        parts = [raw]

    deduped = list(dict.fromkeys(parts))
    if len(deduped) == 1:
        if len(raw) > 80:
            mid = len(raw) // 2
            split_at = raw.rfind(" ", 0, mid)
            if split_at > 20:
                left, right = raw[:split_at].strip(), raw[split_at:].strip()
                if left and right:
                    deduped = [left, right]
        if len(deduped) == 1:
            return expand_research_angles(raw, max_angles=max_sub_queries)

    return tuple(deduped[: max(1, max_sub_queries)])
