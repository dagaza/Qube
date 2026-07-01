"""Adapter catalog — metadata for user-configurable knowledge sources."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.types import (
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
)


@dataclass(frozen=True)
class AdapterCatalogEntry:
    id: str
    label: str
    knowledge_service: str
    ui_group: str
    implemented: bool = False
    requires_api_key: bool = False
    default_enabled: bool = True


ADAPTER_CATALOG: tuple[AdapterCatalogEntry, ...] = (
    # Scientific literature — general science
    AdapterCatalogEntry(
        "pubmed",
        "PubMed",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "crossref",
        "Crossref",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=False,
    ),
    AdapterCatalogEntry(
        "semantic_scholar",
        "Semantic Scholar",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=False,
        requires_api_key=True,
    ),
    AdapterCatalogEntry(
        "arxiv",
        "arXiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "inspire_hep",
        "INSPIRE-HEP",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "nasa_ads",
        "NASA ADS",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=False,
        requires_api_key=True,
        default_enabled=False,
    ),
    # Biology / life sciences (Phase 6c-1)
    AdapterCatalogEntry(
        "pubmed",
        "PubMed",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "biorxiv",
        "bioRxiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
    ),
    # Chemistry (Phase 6c-2)
    AdapterCatalogEntry(
        "pubchem",
        "PubChem",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "pubmed",
        "PubMed",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
    ),
    # Computer science (shared adapters appear in multiple UI groups)
    AdapterCatalogEntry(
        "arxiv",
        "arXiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "dblp",
        "DBLP",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
    ),
    # Economics
    AdapterCatalogEntry(
        "repec",
        "RePEc",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "ssrn",
        "SSRN",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=False,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=True,
    ),
    # Psychology (Phase 6c-3)
    AdapterCatalogEntry(
        "pubmed",
        "PubMed",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Psychology",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Psychology",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "psycinfo",
        "PsycINFO",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Psychology",
        implemented=False,
    ),
    # Social science — sociology & political science (Phase 6c-3)
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "socarxiv",
        "SocArXiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=False,
    ),
    AdapterCatalogEntry(
        "ssrn",
        "SSRN",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=False,
    ),
    # Finance
    AdapterCatalogEntry(
        "sec_edgar",
        "SEC EDGAR",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "companies_house",
        "Companies House",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=False,
    ),
    AdapterCatalogEntry(
        "bloomberg_api",
        "Bloomberg (API)",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=False,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "alpha_vantage",
        "Alpha Vantage",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=False,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "fred",
        "FRED",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=False,
    ),
    # Legal
    AdapterCatalogEntry(
        "courtlistener",
        "CourtListener",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "eur_lex",
        "EUR-Lex",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=False,
    ),
    AdapterCatalogEntry(
        "canlii",
        "CanLII",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=False,
    ),
    AdapterCatalogEntry(
        "bailii",
        "BAILII",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=False,
    ),
)

_CATALOG_BY_ID: dict[str, AdapterCatalogEntry] = {}
for _entry in ADAPTER_CATALOG:
    if _entry.id not in _CATALOG_BY_ID:
        _CATALOG_BY_ID[_entry.id] = _entry


def get_adapter_entry(adapter_id: str) -> AdapterCatalogEntry | None:
    return _CATALOG_BY_ID.get((adapter_id or "").strip().lower())


def catalog_entries_for_service(service_id: str) -> tuple[AdapterCatalogEntry, ...]:
    sid = (service_id or "").strip().lower()
    return tuple(e for e in ADAPTER_CATALOG if e.knowledge_service == sid)


def ui_groups_for_service(service_id: str) -> tuple[str, ...]:
    groups: list[str] = []
    seen: set[str] = set()
    for entry in catalog_entries_for_service(service_id):
        if entry.ui_group not in seen:
            seen.add(entry.ui_group)
            groups.append(entry.ui_group)
    return tuple(groups)


def catalog_entries_for_ui_group(service_id: str, ui_group: str) -> tuple[AdapterCatalogEntry, ...]:
    sid = (service_id or "").strip().lower()
    group = (ui_group or "").strip()
    return tuple(
        e for e in ADAPTER_CATALOG if e.knowledge_service == sid and e.ui_group == group
    )


def default_enabled_adapter_ids(service_id: str) -> tuple[str, ...]:
    """Unique default-enabled adapter ids for a knowledge service."""
    ids: list[str] = []
    seen: set[str] = set()
    for entry in catalog_entries_for_service(service_id):
        if entry.id in seen:
            continue
        seen.add(entry.id)
        if entry.default_enabled and entry.implemented:
            ids.append(entry.id)
    return tuple(ids)


def implemented_adapter_ids(service_id: str) -> frozenset[str]:
    ids: set[str] = set()
    for entry in catalog_entries_for_service(service_id):
        if entry.implemented:
            ids.add(entry.id)
    return frozenset(ids)


def implemented_adapters_for_ui_group(service_id: str, ui_group: str) -> tuple[str, ...]:
    """Ordered implemented adapter ids for a settings UI group."""
    ids: list[str] = []
    seen: set[str] = set()
    for entry in catalog_entries_for_ui_group(service_id, ui_group):
        if entry.id in seen:
            continue
        seen.add(entry.id)
        if entry.implemented:
            ids.append(entry.id)
    return tuple(ids)


CONFIGURABLE_KNOWLEDGE_SERVICES: tuple[tuple[str, str], ...] = (
    (SERVICE_SCIENTIFIC_EVIDENCE, "Scientific literature"),
    (SERVICE_FINANCE_KNOWLEDGE, "Finance"),
    (SERVICE_LEGAL_KNOWLEDGE, "Legal"),
)
