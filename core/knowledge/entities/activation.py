"""Deterministic component activation for entity resolution (ADR 002)."""

from __future__ import annotations

from core.knowledge.entities.registry import (
    ALWAYS_ON_EXTRACTOR_IDS,
    all_activators,
    all_linkers,
    expand_pack_extractor_ids,
    expand_pack_linker_ids,
    get_extractor,
    get_linker,
    pack_ids_for_service,
)
from core.knowledge.entities.types import ActiveComponents, EntityResolutionContext
from core.knowledge.types import EvidenceObject

_DEFAULT_MAX_EXTRACTORS = 8


def _component_ids_from_enables(enables: tuple[str, ...]) -> tuple[set[str], set[str]]:
    extractors: set[str] = set()
    linkers: set[str] = set()
    for component_id in enables:
        if get_extractor(component_id) is not None:
            extractors.add(component_id)
        elif get_linker(component_id) is not None:
            linkers.add(component_id)
    return extractors, linkers


def _apply_extractor_cap(
    extractor_ids: set[str],
    *,
    max_extractors: int,
) -> tuple[str, ...]:
    always_on = [eid for eid in ALWAYS_ON_EXTRACTOR_IDS if eid in extractor_ids]
    optional = sorted(
        (eid for eid in extractor_ids if eid not in ALWAYS_ON_EXTRACTOR_IDS),
        key=lambda eid: (get_extractor(eid).priority if get_extractor(eid) else 999, eid),
    )
    remaining = max(0, max_extractors - len(always_on))
    capped = always_on + optional[:remaining]
    return tuple(dict.fromkeys(capped))


def _kinds_for_extractors(extractor_ids: tuple[str, ...]) -> set[str]:
    kinds: set[str] = set()
    for extractor_id in extractor_ids:
        extractor = get_extractor(extractor_id)
        if extractor is not None:
            kinds.update(extractor.kinds)
    return kinds


def resolve_active_components(
    ctx: EntityResolutionContext,
    *,
    source: EvidenceObject | None = None,
    max_extractors: int = _DEFAULT_MAX_EXTRACTORS,
) -> ActiveComponents:
    extractor_ids: set[str] = set(ALWAYS_ON_EXTRACTOR_IDS)
    linker_ids: set[str] = set()

    hint_pack_ids = pack_ids_for_service(ctx.knowledge_service)
    extractor_ids.update(expand_pack_extractor_ids(hint_pack_ids))
    linker_ids.update(expand_pack_linker_ids(hint_pack_ids))

    query = ctx.query_resolved or ""
    for activator in all_activators():
        matched = activator.matches_query(query)
        if source is not None and activator.matches_source(source):
            matched = True
        if not matched:
            continue
        enabled_extractors, enabled_linkers = _component_ids_from_enables(activator.enables)
        extractor_ids.update(enabled_extractors)
        linker_ids.update(enabled_linkers)

    capped_extractors = _apply_extractor_cap(extractor_ids, max_extractors=max_extractors)
    kinds = _kinds_for_extractors(capped_extractors)

    active_linkers: list[str] = []
    for linker in sorted(all_linkers(), key=lambda link: (link.priority, link.id)):
        if linker.id not in linker_ids:
            continue
        if not any(kind in kinds for kind in linker.input_kinds):
            continue
        active_linkers.append(linker.id)

    return ActiveComponents(
        extractor_ids=capped_extractors,
        linker_ids=tuple(dict.fromkeys(active_linkers)),
    )
