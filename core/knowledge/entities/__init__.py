"""Entity resolution for evidence bundles (Phase 6 Slice 3, ADR 002)."""

from core.knowledge.entities.enrich import (
    enrich_bundle,
    enrich_evidence_object,
    format_entity_labels,
)
from core.knowledge.entities.pipeline import (
    collect_bundle_entity_ids,
    context_from_bundle,
    resolve_entity_ids,
    resolve_entities_for_source,
    resolve_entities_from_text,
)
from core.knowledge.entities.policy import dedupe_cluster_entity_ids
from core.knowledge.entities.types import (
    ActiveComponents,
    EntityResolutionContext,
)

__all__ = [
    "ActiveComponents",
    "EntityResolutionContext",
    "collect_bundle_entity_ids",
    "context_from_bundle",
    "dedupe_cluster_entity_ids",
    "enrich_bundle",
    "enrich_evidence_object",
    "format_entity_labels",
    "resolve_entity_ids",
    "resolve_entities_for_source",
    "resolve_entities_from_text",
]
