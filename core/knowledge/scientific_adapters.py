"""Scientific service adapter selection policy (discipline routing + user prefs)."""

from __future__ import annotations

from core.knowledge.adapters.arxiv_api import ADAPTER_ID as ARXIV_ID
from core.knowledge.adapters.openalex import ADAPTER_ID as OPENALEX_ID
from core.knowledge.adapters.pubmed_eutils import ADAPTER_ID as PUBMED_ID
from core.knowledge.scientific_discipline import (
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL,
    detect_scientific_discipline,
    is_medical_query,
    preferred_adapters_for_discipline,
)

_MEDICAL_ADAPTERS = (PUBMED_ID, OPENALEX_ID, ARXIV_ID)
_SCHOLARLY_ADAPTERS = (OPENALEX_ID, ARXIV_ID)


def apply_scientific_adapter_policy(
    enabled: tuple[str, ...],
    *,
    query: str = "",
    medical_query: bool | None = None,
) -> tuple[str, ...]:
    """Filter and order user-enabled adapters by detected query discipline."""
    match = detect_scientific_discipline(query, medical_query=medical_query)
    preferred = preferred_adapters_for_discipline(match.discipline)

    pool = enabled
    if match.discipline != SCIENTIFIC_DISCIPLINE_BIOMEDICAL:
        pool = tuple(aid for aid in enabled if aid != PUBMED_ID)

    ordered = [aid for aid in preferred if aid in pool]
    if ordered:
        return tuple(ordered)

    if match.discipline == SCIENTIFIC_DISCIPLINE_BIOMEDICAL:
        return _fallback_from_enabled(pool, _MEDICAL_ADAPTERS)
    return _fallback_from_enabled(pool, _SCHOLARLY_ADAPTERS)


def default_scientific_adapters_for_query(
    query: str,
    *,
    composer_adapter_filter: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Legacy default when no user preferences are stored."""
    if composer_adapter_filter:
        return composer_adapter_filter
    match = detect_scientific_discipline(query)
    return preferred_adapters_for_discipline(match.discipline)


def _fallback_from_enabled(
    enabled: tuple[str, ...],
    preferred_order: tuple[str, ...],
) -> tuple[str, ...]:
    ordered = [aid for aid in preferred_order if aid in enabled]
    if ordered:
        return tuple(ordered)
    return enabled
