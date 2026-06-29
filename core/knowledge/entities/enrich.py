"""Attach resolved entity_ids to evidence objects and bundles."""

from __future__ import annotations

from dataclasses import replace

from core.app_settings import entity_resolution_enabled
from core.knowledge.entities.pipeline import (
    context_from_bundle,
    resolve_entities_for_source,
)
from core.knowledge.entities.types import EntityResolutionContext
from core.knowledge.types import EvidenceBundle, EvidenceObject


def enrich_evidence_object(
    source: EvidenceObject,
    ctx: EntityResolutionContext | None = None,
) -> EvidenceObject:
    if not entity_resolution_enabled():
        return source
    if source.entity_ids:
        return source
    entity_ids = resolve_entities_for_source(source, ctx)
    if not entity_ids:
        return source
    return replace(source, entity_ids=entity_ids)


def enrich_bundle(
    bundle: EvidenceBundle,
    ctx: EntityResolutionContext | None = None,
) -> EvidenceBundle:
    if not entity_resolution_enabled() or not bundle.sources:
        return bundle
    resolution_ctx = ctx or context_from_bundle(bundle)
    enriched = tuple(enrich_evidence_object(s, resolution_ctx) for s in bundle.sources)
    if enriched == bundle.sources:
        return bundle
    return replace(bundle, sources=enriched)


def format_entity_labels(entity_ids: tuple[str, ...], *, max_labels: int = 8) -> list[str]:
    labels: list[str] = []
    for eid in entity_ids:
        parts = str(eid).split(":", 2)
        if len(parts) < 3:
            continue
        kind = parts[1].replace("_", " ")
        name = parts[2].replace("-", " ")
        labels.append(f"{name} ({kind})")
        if len(labels) >= max_labels:
            break
    return labels


def format_entity_id_label(entity_id: str) -> str:
    parts = str(entity_id).split(":", 2)
    if len(parts) < 3:
        return entity_id
    return f"{parts[2].replace('-', ' ')} ({parts[1].replace('_', ' ')})"
