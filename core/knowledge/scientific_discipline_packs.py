"""Scientific discipline pack registry (Phase 6c foundation).

Discipline packs live under ``scientific_evidence`` only. Finance and Legal are
separate Knowledge Services — see ``docs/phase6c_scientific_discipline_packs.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

DisciplineStatus = Literal["active", "stub", "planned"]

# Backward-compatible ids from Slice 6a (biomedical → medicine alias at detection layer).
SCIENTIFIC_DISCIPLINE_MEDICINE = "medicine"
SCIENTIFIC_DISCIPLINE_BIOLOGY = "biology"
SCIENTIFIC_DISCIPLINE_BIOMEDICAL = "biomedical"  # alias for medicine
SCIENTIFIC_DISCIPLINE_CHEMISTRY = "chemistry"
SCIENTIFIC_DISCIPLINE_PHYSICS = "physics"
SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE = "computer_science"
SCIENTIFIC_DISCIPLINE_ECONOMICS = "economics"
SCIENTIFIC_DISCIPLINE_PSYCHOLOGY = "psychology"
SCIENTIFIC_DISCIPLINE_SOCIOLOGY = "sociology"
SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE = "political_science"
SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT = "earth_environment"
SCIENTIFIC_DISCIPLINE_GENERAL = "general_science"

DISCIPLINE_PACK_VERSION = "1.0.0"

_UNIVERSAL_FALLBACK_ADAPTERS: tuple[str, ...] = ("openalex",)


@dataclass(frozen=True)
class ScientificDisciplinePack:
    """Routing metadata for one scholarly discipline within scientific_evidence."""

    id: str
    label: str
    ui_group: str
    primary_adapters: tuple[str, ...]
    fallback_adapters: tuple[str, ...] = _UNIVERSAL_FALLBACK_ADAPTERS
    entity_pack_hints: tuple[str, ...] = ()
    status: DisciplineStatus = "planned"
    notes: str = ""

    @property
    def knowledge_service(self) -> str:
        return SERVICE_SCIENTIFIC_EVIDENCE

    def resolved_adapter_order(self) -> tuple[str, ...]:
        """Primary adapters followed by fallbacks (deduped, order preserved)."""
        seen: set[str] = set()
        ordered: list[str] = []
        for adapter_id in (*self.primary_adapters, *self.fallback_adapters):
            if adapter_id in seen:
                continue
            seen.add(adapter_id)
            ordered.append(adapter_id)
        return tuple(ordered)


SCIENTIFIC_DISCIPLINE_PACKS: tuple[ScientificDisciplinePack, ...] = (
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_MEDICINE,
        label="Medicine",
        ui_group="Science",
        primary_adapters=("pubmed",),
        fallback_adapters=("openalex", "pubmed"),
        entity_pack_hints=("biomedical",),
        status="active",
        notes="Clinical and therapeutic queries; maps from legacy biomedical id.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_BIOLOGY,
        label="Biology",
        ui_group="Biology",
        primary_adapters=("pubmed", "biorxiv"),
        fallback_adapters=("openalex",),
        entity_pack_hints=("biomedical",),
        status="active",
        notes="6c-1: molecular, ecological, evolutionary life science; bioRxiv stub + Europe PMC.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_CHEMISTRY,
        label="Chemistry",
        ui_group="Chemistry",
        primary_adapters=("pubchem",),
        fallback_adapters=("openalex", "pubmed"),
        status="active",
        notes="6c-2: PubChem PUG REST for compounds and properties.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_PHYSICS,
        label="Physics",
        ui_group="Science",
        primary_adapters=("arxiv", "inspire_hep"),
        fallback_adapters=("openalex",),
        status="active",
        notes="6c-5: arXiv + open INSPIRE-HEP; NASA ADS catalog stub (API key).",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
        label="Computer Science",
        ui_group="Computer Science",
        primary_adapters=("arxiv", "dblp"),
        fallback_adapters=("openalex",),
        status="active",
        notes="6b: DBLP live.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_ECONOMICS,
        label="Economics",
        ui_group="Economics",
        primary_adapters=("repec",),
        fallback_adapters=("openalex", "ssrn"),
        status="active",
        notes="6c-4: RePEc live via EconBiz API; SSRN catalog stub.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_PSYCHOLOGY,
        label="Psychology",
        ui_group="Psychology",
        primary_adapters=("pubmed",),
        fallback_adapters=("openalex",),
        status="active",
        notes="6c-3: cognitive/experimental psychology; PubMed + OpenAlex.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_SOCIOLOGY,
        label="Sociology",
        ui_group="Social Science",
        primary_adapters=("openalex",),
        fallback_adapters=("socarxiv",),
        status="active",
        notes="6c-3: OpenAlex primary; SocArXiv stub planned.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE,
        label="Political Science",
        ui_group="Social Science",
        primary_adapters=("openalex",),
        fallback_adapters=("ssrn",),
        status="active",
        notes="6c-3: OpenAlex primary; SSRN stub shared with economics catalog.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT,
        label="Earth & Environment",
        ui_group="Science",
        primary_adapters=("openalex",),
        fallback_adapters=("arxiv",),
        status="planned",
        notes="Climate, geoscience; successor to many general_science geo queries.",
    ),
    ScientificDisciplinePack(
        id=SCIENTIFIC_DISCIPLINE_GENERAL,
        label="General science",
        ui_group="Science",
        primary_adapters=("openalex",),
        fallback_adapters=("arxiv", "pubmed"),
        status="active",
        notes="Interdisciplinary fallback when no discipline signal matches.",
    ),
)

_PACK_BY_ID: dict[str, ScientificDisciplinePack] = {p.id: p for p in SCIENTIFIC_DISCIPLINE_PACKS}

# Legacy detection id → canonical pack id
_DISCIPLINE_ALIASES: dict[str, str] = {
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL: SCIENTIFIC_DISCIPLINE_MEDICINE,
}


def normalize_discipline_id(discipline_id: str) -> str:
    key = (discipline_id or "").strip().lower()
    return _DISCIPLINE_ALIASES.get(key, key)


def get_discipline_pack(discipline_id: str) -> ScientificDisciplinePack | None:
    return _PACK_BY_ID.get(normalize_discipline_id(discipline_id))


def discipline_packs_for_service(service_id: str) -> tuple[ScientificDisciplinePack, ...]:
    sid = (service_id or "").strip().lower()
    return tuple(p for p in SCIENTIFIC_DISCIPLINE_PACKS if p.knowledge_service == sid)


def active_discipline_ids() -> tuple[str, ...]:
    return tuple(p.id for p in SCIENTIFIC_DISCIPLINE_PACKS if p.status == "active")


def planned_primary_adapter_ids() -> frozenset[str]:
    """All primary adapter ids referenced by any pack (for catalog gap analysis)."""
    ids: set[str] = set()
    for pack in SCIENTIFIC_DISCIPLINE_PACKS:
        ids.update(pack.primary_adapters)
        ids.update(pack.fallback_adapters)
    return frozenset(ids)
