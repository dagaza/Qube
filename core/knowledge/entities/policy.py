"""Central entity kind policy for merge dedupe (ADR 002)."""

from __future__ import annotations

from core.knowledge.entities.ids import entity_kind

ENTITY_KIND_POLICY: dict[str, str] = {
    "doi": "work_id",
    "pubmed": "work_id",
    "arxiv": "work_id",
    "isbn": "work_id",
    "trial": "work_id",
    "drug": "work_id",
    "rxnorm": "work_id",
    "drug_class": "concept",
    "condition": "concept",
    "topic": "concept",
}


def is_dedupe_cluster_entity(entity_id: str) -> bool:
    """Entity ids that identify a specific work, not a broad concept class."""
    return ENTITY_KIND_POLICY.get(entity_kind(entity_id)) == "work_id"


def dedupe_cluster_entity_ids(entity_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Subset of entity ids suitable for same-work clustering during merge dedupe."""
    return tuple(sorted(e for e in entity_ids if is_dedupe_cluster_entity(e)))
