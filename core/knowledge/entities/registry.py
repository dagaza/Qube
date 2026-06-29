"""Component and pack registry for entity resolution (ADR 002)."""

from __future__ import annotations

from core.knowledge.entities.activators.biomedical import BIOMEDICAL_ACTIVATOR
from core.knowledge.entities.extractors.bibliographic import BIBLIOGRAPHIC_EXTRACTOR
from core.knowledge.entities.extractors.biomedical_conditions import (
    BIOMEDICAL_CONDITIONS_EXTRACTOR,
)
from core.knowledge.entities.extractors.biomedical_drugs import BIOMEDICAL_DRUGS_EXTRACTOR
from core.knowledge.entities.extractors.biomedical_trials import BIOMEDICAL_TRIALS_EXTRACTOR
from core.knowledge.entities.linkers.rxnorm import RXNORM_LINKER
from core.knowledge.entities.packs.definitions import PACK_DEFINITIONS
from core.knowledge.entities.types import (
    EntityActivator,
    EntityExtractor,
    EntityLinker,
    EntityPackDefinition,
)
from core.knowledge.types import (
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_GENERAL_WEB,
    SERVICE_INTERNAL_CORPUS,
    SERVICE_SCIENTIFIC_EVIDENCE,
)

ALWAYS_ON_EXTRACTOR_IDS: tuple[str, ...] = ("bibliographic",)

ENTITY_PACK_HINTS: dict[str, tuple[str, ...]] = {
    SERVICE_SCIENTIFIC_EVIDENCE: ("bibliographic",),
    SERVICE_GENERAL_WEB: ("bibliographic",),
    SERVICE_INTERNAL_CORPUS: ("bibliographic",),
    SERVICE_FINANCE_KNOWLEDGE: ("bibliographic",),
}

_EXTRACTORS: dict[str, EntityExtractor] = {
    BIBLIOGRAPHIC_EXTRACTOR.id: BIBLIOGRAPHIC_EXTRACTOR,
    BIOMEDICAL_DRUGS_EXTRACTOR.id: BIOMEDICAL_DRUGS_EXTRACTOR,
    BIOMEDICAL_CONDITIONS_EXTRACTOR.id: BIOMEDICAL_CONDITIONS_EXTRACTOR,
    BIOMEDICAL_TRIALS_EXTRACTOR.id: BIOMEDICAL_TRIALS_EXTRACTOR,
}

_ACTIVATORS: dict[str, EntityActivator] = {
    BIOMEDICAL_ACTIVATOR.id: BIOMEDICAL_ACTIVATOR,
}

_LINKERS: dict[str, EntityLinker] = {
    RXNORM_LINKER.id: RXNORM_LINKER,
}

_PACKS: dict[str, EntityPackDefinition] = {pack.id: pack for pack in PACK_DEFINITIONS}


def get_extractor(extractor_id: str) -> EntityExtractor | None:
    return _EXTRACTORS.get(extractor_id)


def get_activator(activator_id: str) -> EntityActivator | None:
    return _ACTIVATORS.get(activator_id)


def get_linker(linker_id: str) -> EntityLinker | None:
    return _LINKERS.get(linker_id)


def get_pack(pack_id: str) -> EntityPackDefinition | None:
    return _PACKS.get(pack_id)


def all_packs() -> tuple[EntityPackDefinition, ...]:
    return PACK_DEFINITIONS


def all_extractors() -> tuple[EntityExtractor, ...]:
    return tuple(_EXTRACTORS[id_] for id_ in sorted(_EXTRACTORS))


def all_activators() -> tuple[EntityActivator, ...]:
    return tuple(_ACTIVATORS[id_] for id_ in sorted(_ACTIVATORS))


def all_linkers() -> tuple[EntityLinker, ...]:
    return tuple(_LINKERS[id_] for id_ in sorted(_LINKERS))


def pack_ids_for_service(service_id: str) -> tuple[str, ...]:
    return ENTITY_PACK_HINTS.get((service_id or SERVICE_GENERAL_WEB).strip().lower(), ())


def expand_pack_extractor_ids(pack_ids: tuple[str, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for pack_id in pack_ids:
        pack = get_pack(pack_id)
        if pack is None:
            continue
        for extractor_id in pack.extractor_ids:
            if extractor_id not in seen:
                seen.add(extractor_id)
                ids.append(extractor_id)
    return tuple(ids)


def expand_pack_linker_ids(pack_ids: tuple[str, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for pack_id in pack_ids:
        pack = get_pack(pack_id)
        if pack is None:
            continue
        for linker_id in pack.linker_ids:
            if linker_id not in seen:
                seen.add(linker_id)
                ids.append(linker_id)
    return tuple(ids)
