"""Entity pack definitions (registration bundles only)."""

from __future__ import annotations

from core.knowledge.entities.types import EntityPackDefinition
from core.knowledge.types import (
    SERVICE_GENERAL_WEB,
    SERVICE_INTERNAL_CORPUS,
    SERVICE_SCIENTIFIC_EVIDENCE,
)

BIBLIOGRAPHIC_PACK = EntityPackDefinition(
    id="bibliographic",
    extractor_ids=("bibliographic",),
    activator_ids=(),
    linker_ids=(),
    service_hint_services=frozenset(
        {SERVICE_GENERAL_WEB, SERVICE_SCIENTIFIC_EVIDENCE, SERVICE_INTERNAL_CORPUS}
    ),
)

BIOMEDICAL_PACK = EntityPackDefinition(
    id="biomedical",
    extractor_ids=(
        "biomedical_drugs",
        "biomedical_conditions",
        "biomedical_trials",
    ),
    activator_ids=("biomedical",),
    linker_ids=("rxnorm",),
    service_hint_services=frozenset({SERVICE_SCIENTIFIC_EVIDENCE}),
)

PACK_DEFINITIONS: tuple[EntityPackDefinition, ...] = (
    BIBLIOGRAPHIC_PACK,
    BIOMEDICAL_PACK,
)
