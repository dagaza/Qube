"""Backward-compatible resolve exports (delegates to pipeline + policy)."""

from __future__ import annotations

from core.knowledge.entities.pipeline import (
    collect_bundle_entity_ids,
    resolve_entities_for_source,
    resolve_entities_from_text,
)
from core.knowledge.entities.policy import dedupe_cluster_entity_ids

__all__ = [
    "collect_bundle_entity_ids",
    "dedupe_cluster_entity_ids",
    "resolve_entities_for_source",
    "resolve_entities_from_text",
]
