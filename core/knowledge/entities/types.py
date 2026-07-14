"""Entity resolution types and protocols (ADR 002).

Stable contract: pipeline output is canonical ``entity_ids`` on ``EvidenceObject``.

Planned evolution (ADR 003 — not implemented): extractors may later emit
``EntityOccurrence`` records, with a separate ``EntityNormalizer`` stage
producing the same canonical ids. Activators, linkers, packs, and the
activation engine are intended to remain unchanged. See
``docs/external_knowledge_platform_plan.md`` §21 Future evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from core.knowledge.types import EvidenceObject

CostTier = Literal["cheap", "expensive"]


@dataclass(frozen=True)
class EntityResolutionContext:
    query_resolved: str
    knowledge_service: str
    composer_tool: str | None = None
    adapter_filter: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ActiveComponents:
    extractor_ids: tuple[str, ...]
    linker_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityPackDefinition:
    id: str
    extractor_ids: tuple[str, ...]
    activator_ids: tuple[str, ...]
    linker_ids: tuple[str, ...] = ()
    service_hint_services: frozenset[str] = frozenset()


class EntityExtractor(Protocol):
    """Detect mentions in text and return canonical entity ids (v1).

    v1: ``extract()`` returns final ``entity:{kind}:{key}`` strings.
    Keep surface detection and ``make_entity_id`` construction in separate
    functions within the extractor module so they can split when ADR 003
    adds ``extract_occurrences()`` + ``EntityNormalizer`` without a
    registry rewrite.
    """

    id: str
    pack_id: str
    kinds: tuple[str, ...]
    priority: int
    cost: CostTier

    def extract(self, text: str, *, doi: str | None = None) -> tuple[str, ...]: ...


class EntityActivator(Protocol):
    id: str
    pack_id: str
    priority: int
    enables: tuple[str, ...]

    def matches_query(self, query: str) -> bool: ...

    def matches_source(self, source: EvidenceObject) -> bool: ...


class EntityLinker(Protocol):
    id: str
    pack_id: str
    input_kinds: tuple[str, ...]
    priority: int
    requires_network: bool

    def link(self, entity_ids: tuple[str, ...]) -> tuple[str, ...]: ...
