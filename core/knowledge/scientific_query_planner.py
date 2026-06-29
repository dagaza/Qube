"""Deterministic query planning for scientific_evidence (Stage 1).

Maps conversational @evidence / @science prompts to adapter-oriented keyword queries
while leaving semantic_query unchanged for OpenAlex ranking/embeddings.

The scientific service spans all scholarly disciplines (medicine, CS, physics,
economics, etc.). Medical entity keyword extraction here is a **medical discipline
helper** only — activated when the biomedical activator matches the query. It is not
the service boundary. Stage 2 will add discipline detection and adapter routing;
see platform plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.entities.activators.biomedical import BIOMEDICAL_ACTIVATOR
from core.knowledge.entities.conditions import extract_condition_entities
from core.knowledge.entities.drug_classes import extract_drug_entities
from core.knowledge.entities.trials import extract_trial_entities

_CONVERSATIONAL_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^summarize(?:\s+key\s+outcomes\s+from)?\s+(?:the\s+)?",
        re.IGNORECASE,
    ),
    re.compile(r"^what\s+does\s+the\s+literature\s+say\s+about\s+", re.IGNORECASE),
    re.compile(r"^what\s+is\s+the\s+evidence\s+for\s+", re.IGNORECASE),
    re.compile(r"^what\s+do\s+studies\s+show\s+(?:about\s+)?", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^explain\s+", re.IGNORECASE),
    re.compile(r"^describe\s+", re.IGNORECASE),
    re.compile(r"^overview\s+of\s+", re.IGNORECASE),
)

_TRAILING_FILLER = re.compile(
    r"\s+(?:please|thanks|thank\s+you)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScientificQueryPlan:
    keyword_query: str
    semantic_query: str
    entity_keywords: tuple[str, ...] = ()


def _strip_conversational_phrasing(text: str) -> str:
    q = _TRAILING_FILLER.sub("", (text or "").strip())
    for pattern in _CONVERSATIONAL_PREFIXES:
        q = pattern.sub("", q).strip()
    q = re.sub(r"^\s*(?:the\s+)", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s+trial\s+(?:for|in)\s+", " ", q, flags=re.IGNORECASE).strip()
    return sanitize_api_query(q)


def _extract_entity_keyword_labels(text: str) -> tuple[str, ...]:
    """Medical discipline keyword labels for PubMed/arXiv — not general entity resolution."""
    if not BIOMEDICAL_ACTIVATOR.matches_query(text):
        return ()
    labels: list[str] = []
    for _eid, label in extract_trial_entities(text):
        labels.append(label)
    for _eid, label in extract_drug_entities(text):
        labels.append(label)
    for _eid, label in extract_condition_entities(text):
        labels.append(label)
    return tuple(dict.fromkeys(part for part in labels if part))


def _build_keyword_query(
    *,
    stripped: str,
    entity_keywords: tuple[str, ...],
    fallback: str,
) -> str:
    if entity_keywords:
        return " ".join(entity_keywords)
    if stripped and len(stripped.split()) >= 2:
        return stripped
    return fallback


def plan_scientific_query(
    query: str,
    *,
    semantic_query: str | None = None,
) -> ScientificQueryPlan:
    """Produce adapter-oriented queries for scientific_evidence retrieval."""
    raw = sanitize_api_query(query)
    semantic = sanitize_api_query(semantic_query or query)
    entity_keywords = _extract_entity_keyword_labels(raw)
    stripped = _strip_conversational_phrasing(raw)
    keyword = _build_keyword_query(
        stripped=stripped,
        entity_keywords=entity_keywords,
        fallback=semantic or raw,
    )
    return ScientificQueryPlan(
        keyword_query=keyword,
        semantic_query=semantic,
        entity_keywords=entity_keywords,
    )


def adapter_query_for(plan: ScientificQueryPlan, adapter_id: str) -> str:
    """PubMed/arXiv use keywords; OpenAlex keeps semantic query."""
    if adapter_id in {"pubmed", "arxiv"}:
        return plan.keyword_query
    return plan.semantic_query
