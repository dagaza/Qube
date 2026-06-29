"""Scientific service adapter selection policy (medical gating + user prefs)."""

from __future__ import annotations

import re

from core.knowledge.adapters.arxiv_api import ADAPTER_ID as ARXIV_ID
from core.knowledge.adapters.openalex import ADAPTER_ID as OPENALEX_ID
from core.knowledge.adapters.pubmed_eutils import ADAPTER_ID as PUBMED_ID
from core.knowledge.entities.activators.biomedical import BIOMEDICAL_ACTIVATOR

_MEDICAL_ADAPTERS = (PUBMED_ID, OPENALEX_ID, ARXIV_ID)
_SCHOLARLY_ADAPTERS = (OPENALEX_ID, ARXIV_ID)

_MEDICAL_HINTS = re.compile(
    r"\b(drug|medication|medicine|disease|symptom|treatment|clinical|patient|"
    r"therapy|diagnosis|fda|vaccine|diabetes|cancer|ozempic|semaglutide)\b",
    re.IGNORECASE,
)


def is_medical_query(query: str) -> bool:
    text = query or ""
    return bool(BIOMEDICAL_ACTIVATOR.matches_query(text) or _MEDICAL_HINTS.search(text))


def apply_scientific_adapter_policy(
    enabled: tuple[str, ...],
    *,
    query: str = "",
    medical_query: bool | None = None,
) -> tuple[str, ...]:
    """Filter user-enabled scientific adapters by query discipline (medical vs general)."""
    is_medical = is_medical_query(query) if medical_query is None else bool(medical_query)
    if is_medical:
        ordered = [aid for aid in _MEDICAL_ADAPTERS if aid in enabled]
        return tuple(ordered) if ordered else _fallback_from_enabled(enabled, _MEDICAL_ADAPTERS)

    without_pubmed = tuple(aid for aid in enabled if aid != PUBMED_ID)
    if without_pubmed:
        return without_pubmed
    return _fallback_from_enabled(enabled, _SCHOLARLY_ADAPTERS)


def default_scientific_adapters_for_query(
    query: str,
    *,
    composer_adapter_filter: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Legacy default when no user preferences are stored."""
    if composer_adapter_filter:
        return composer_adapter_filter
    if is_medical_query(query):
        return _MEDICAL_ADAPTERS
    return _SCHOLARLY_ADAPTERS


def _fallback_from_enabled(
    enabled: tuple[str, ...],
    preferred_order: tuple[str, ...],
) -> tuple[str, ...]:
    ordered = [aid for aid in preferred_order if aid in enabled]
    if ordered:
        return tuple(ordered)
    return enabled
