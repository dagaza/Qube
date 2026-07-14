"""Entity resolution pipeline orchestration (ADR 002).

Insertion point for ADR 003 (planned): after extractors, before linkers —
``EntityNormalizer.normalize(extract_occurrences(...))`` producing the same
canonical ``entity_ids`` attached to ``EvidenceObject``.
"""

from __future__ import annotations

from core.knowledge.entities.activation import resolve_active_components
from core.knowledge.entities.registry import get_extractor, get_linker
from core.knowledge.entities.types import EntityResolutionContext
from core.knowledge.types import EvidenceBundle, EvidenceObject
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE


def context_from_bundle(
    bundle: EvidenceBundle,
    *,
    composer_tool: str | None = None,
    adapter_filter: tuple[str, ...] | None = None,
) -> EntityResolutionContext:
    return EntityResolutionContext(
        query_resolved=bundle.query_resolved or bundle.query_raw,
        knowledge_service=bundle.knowledge_service,
        composer_tool=composer_tool,
        adapter_filter=adapter_filter,
    )


def _default_context_for_text(text: str) -> EntityResolutionContext:
    return EntityResolutionContext(
        query_resolved=text,
        knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
    )


def resolve_entity_ids(
    text: str,
    ctx: EntityResolutionContext,
    *,
    source: EvidenceObject | None = None,
    doi: str | None = None,
) -> tuple[str, ...]:
    active = resolve_active_components(ctx, source=source)
    ids: set[str] = set()
    for extractor_id in active.extractor_ids:
        extractor = get_extractor(extractor_id)
        if extractor is None:
            continue
        ids.update(extractor.extract(text, doi=doi))
    sorted_ids = tuple(sorted(ids))
    for linker_id in active.linker_ids:
        linker = get_linker(linker_id)
        if linker is None:
            continue
        linked = linker.link(sorted_ids)
        if linked:
            ids.update(linked)
    return tuple(sorted(ids))


def resolve_entities_from_text(
    text: str,
    *,
    doi: str | None = None,
    ctx: EntityResolutionContext | None = None,
) -> tuple[str, ...]:
    """Return stable entity ids detected in free text (offline heuristics)."""
    resolution_ctx = ctx or _default_context_for_text(text)
    return resolve_entity_ids(text, resolution_ctx, doi=doi)


def resolve_entities_for_source(
    source: EvidenceObject,
    ctx: EntityResolutionContext | None = None,
) -> tuple[str, ...]:
    blob = " ".join(
        part
        for part in (
            source.title,
            source.excerpt,
            source.full_text or "",
            source.url or "",
            source.source_id or "",
        )
        if part
    )
    resolution_ctx = ctx or EntityResolutionContext(
        query_resolved="",
        knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
    )
    return resolve_entity_ids(blob, resolution_ctx, source=source, doi=source.doi)


def collect_bundle_entity_ids(
    bundle: EvidenceBundle,
    ctx: EntityResolutionContext | None = None,
) -> tuple[str, ...]:
    resolution_ctx = ctx or context_from_bundle(bundle)
    keys: set[str] = set()
    keys.update(
        resolve_entity_ids(
            bundle.query_resolved or bundle.query_raw,
            resolution_ctx,
        )
    )
    for source in bundle.sources:
        if source.entity_ids:
            keys.update(source.entity_ids)
        else:
            keys.update(resolve_entities_for_source(source, resolution_ctx))
    return tuple(sorted(keys))
