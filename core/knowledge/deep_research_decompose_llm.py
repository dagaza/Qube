"""Optional LLM sub-query decomposition for deep research (Phase 5 slice 5)."""

from __future__ import annotations

import json
import re
from typing import Callable, Sequence

from core.knowledge.deep_research_decompose import (
    MAX_SUB_QUERIES,
    decompose_query,
    normalize_deep_research_query,
)

_DECOMPOSE_MAX_TOKENS = 180
_DECOMPOSE_TEMPERATURE = 0.15

_SYSTEM_PROMPT = (
    "You are Qube's deep-research planner. Given a research question, produce "
    "2–3 distinct PubMed-style search queries covering different evidence angles "
    "(e.g. randomized trials, meta-analyses, drug-class names, mortality outcomes).\n\n"
    "Return STRICT JSON only:\n"
    '{"sub_queries": ["query one", "query two"]}\n\n'
    "Rules:\n"
    "- Each query 12–120 characters, English, suitable for scientific indexes\n"
    "- Preserve the user's core topic across queries\n"
    "- No duplicate or near-duplicate queries\n"
    "- Maximum 3 queries"
)


def build_decompose_user_prompt(query: str) -> str:
    return f"Research question:\n{normalize_deep_research_query(query)}"


def _extract_json_object(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_llm_sub_queries(raw: str) -> list[str]:
    """Parse sub-queries from LLM JSON or numbered-list output."""
    parsed = _extract_json_object(raw)
    if parsed is not None:
        candidates = parsed.get("sub_queries") or parsed.get("queries") or []
        if isinstance(candidates, list):
            return [str(q).strip() for q in candidates if str(q).strip()]

    lines: list[str] = []
    for line in (raw or "").splitlines():
        stripped = re.sub(r"^\s*(?:\d+[\).\]]\s*|[-*]\s*)", "", line.strip())
        if len(stripped) >= 12:
            lines.append(stripped)
    return lines


def validate_llm_sub_queries(
    query: str,
    candidates: Sequence[str],
    *,
    max_sub_queries: int = MAX_SUB_QUERIES,
) -> tuple[str, ...]:
    """Normalize, dedupe, and bound LLM sub-queries."""
    base = normalize_deep_research_query(query)
    if not base:
        return ()

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if len(text) < 12 or len(text) > 160:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    if base.lower() not in seen:
        cleaned.insert(0, base)
        seen.add(base.lower())

    if len(cleaned) < 2:
        return ()
    return tuple(cleaned[: max(1, max_sub_queries)])


def decompose_query_with_llm(
    query: str,
    generate_fn: Callable[[str, str], str],
    *,
    max_sub_queries: int = MAX_SUB_QUERIES,
) -> tuple[str, ...]:
    """LLM decomposition with heuristic fallback."""
    raw = generate_fn(_SYSTEM_PROMPT, build_decompose_user_prompt(query))
    candidates = parse_llm_sub_queries(raw)
    validated = validate_llm_sub_queries(
        query,
        candidates,
        max_sub_queries=max_sub_queries,
    )
    if validated:
        return validated
    return decompose_query(query, max_sub_queries=max_sub_queries)
