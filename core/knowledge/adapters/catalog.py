"""Adapter catalog — metadata for user-configurable knowledge sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.knowledge.adapter_readiness import ReadinessLevel
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
    optional_api_key: bool = False
    default_enabled: bool = True
    readiness: ReadinessLevel | None = None
    production_strategy: str = ""


def readiness_for_entry(entry: AdapterCatalogEntry):
    """Resolved readiness metadata for a catalog row."""
    from core.knowledge.adapter_readiness import readiness_for_catalog_entry

    return readiness_for_catalog_entry(entry)


ADAPTER_CATALOG: tuple[AdapterCatalogEntry, ...] = (
    # Scientific literature — general science
    AdapterCatalogEntry(
        "pubmed",
        "PubMed",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        optional_api_key=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        optional_api_key=True,
    ),
    AdapterCatalogEntry(
        "crossref",
        "Crossref",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "semantic_scholar",
        "Semantic Scholar",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
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
        "europe_pmc",
        "Europe PMC",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "nasa_ads",
        "NASA ADS",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
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
        "uniprot",
        "UniProt",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "pdb",
        "Protein Data Bank",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "europe_pmc",
        "Europe PMC",
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
        optional_api_key=True,
    ),
    AdapterCatalogEntry(
        "chembl",
        "ChEMBL",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "chemrxiv",
        "ChemRxiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "uspto_patentsview",
        "USPTO PatentsView",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "epo_espacenet",
        "EPO Espacenet",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
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
    AdapterCatalogEntry(
        "acm_dl",
        "ACM Digital Library",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
        optional_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "openreview",
        "OpenReview",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "acl_anthology",
        "ACL Anthology",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
        default_enabled=True,
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
        implemented=True,
        default_enabled=False,
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
        "psyarxiv",
        "PsyArXiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Psychology",
        implemented=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "psycinfo",
        "PsycINFO",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Psychology",
        implemented=True,
        requires_api_key=True,
        default_enabled=False,
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
        implemented=True,
    ),
    AdapterCatalogEntry(
        "ssrn",
        "SSRN",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=True,
        default_enabled=False,
    ),
    # Earth & environment
    AdapterCatalogEntry(
        "openalex",
        "OpenAlex",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "arxiv",
        "arXiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "noaa",
        "NOAA NCEI",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "nasa_earthdata",
        "NASA Earthdata",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
        default_enabled=False,
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
        implemented=True,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "bloomberg_api",
        "Bloomberg (API)",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "alpha_vantage",
        "Alpha Vantage",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "fred",
        "FRED",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "world_bank",
        "World Bank Open Data",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "eurostat",
        "Eurostat",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "bls",
        "BLS",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    # P0 institutional sources — health & guidelines
    AdapterCatalogEntry(
        "clinicaltrials_gov",
        "ClinicalTrials.gov",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "openfda",
        "openFDA",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "clinicaltrials_gov",
        "ClinicalTrials.gov",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "openfda",
        "openFDA",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Chemistry",
        implemented=True,
        default_enabled=True,
    ),
    # P0 — official statistics (scientific routing)
    AdapterCatalogEntry(
        "world_bank",
        "World Bank Open Data",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "eurostat",
        "Eurostat",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "bls",
        "BLS",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "world_bank",
        "World Bank Open Data",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "eurostat",
        "Eurostat",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "us_census",
        "U.S. Census Bureau",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=True,
        optional_api_key=True,
        default_enabled=True,
    ),
    # P0 — Earth & geoscience
    AdapterCatalogEntry(
        "usgs",
        "USGS Publications",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
        default_enabled=True,
    ),
    # P0 — Agriculture & nutrition
    AdapterCatalogEntry(
        "usda_fdc",
        "USDA FoodData Central",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Agriculture & Nutrition",
        implemented=True,
        optional_api_key=True,
        default_enabled=True,
    ),
    # P0 — Engineering
    AdapterCatalogEntry(
        "ieee_xplore",
        "IEEE Xplore",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Engineering",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "nist",
        "NIST NVD",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Engineering",
        implemented=True,
        optional_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "ietf_rfc",
        "IETF RFCs",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Engineering",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "uspto_patentsview",
        "USPTO PatentsView",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Engineering",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "epo_espacenet",
        "EPO Espacenet",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Engineering",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "arxiv",
        "arXiv",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Engineering",
        implemented=True,
    ),
    AdapterCatalogEntry(
        "ieee_xplore",
        "IEEE Xplore",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "ietf_rfc",
        "IETF RFCs",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Computer Science",
        implemented=True,
        default_enabled=True,
    ),
    # Slice 13 — health guidelines & official statistics
    AdapterCatalogEntry(
        "nice",
        "NICE",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "cdc",
        "CDC",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "who",
        "WHO GHO",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "nice",
        "NICE",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "cdc",
        "CDC",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "who",
        "WHO GHO",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Biology",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "oecd",
        "OECD",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Economics",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "oecd",
        "OECD",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Social Science",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "oecd",
        "OECD",
        SERVICE_FINANCE_KNOWLEDGE,
        "Finance",
        implemented=True,
        default_enabled=True,
    ),
    # Slice 14 — geoscience & agriculture
    AdapterCatalogEntry(
        "ipcc",
        "IPCC",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "copernicus_cds",
        "Copernicus CDS",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Earth & Environment",
        implemented=True,
        optional_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "fao",
        "FAOSTAT",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Agriculture & Nutrition",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "usda",
        "USDA ERS",
        SERVICE_SCIENTIFIC_EVIDENCE,
        "Agriculture & Nutrition",
        implemented=True,
        optional_api_key=True,
        default_enabled=True,
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
        implemented=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "canlii",
        "CanLII",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=True,
        requires_api_key=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "bailii",
        "BAILII",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=True,
        default_enabled=False,
    ),
    AdapterCatalogEntry(
        "congress_gov",
        "Congress.gov",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "govinfo",
        "GovInfo",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=True,
        requires_api_key=True,
        default_enabled=True,
    ),
    AdapterCatalogEntry(
        "legislation_uk",
        "legislation.gov.uk",
        SERVICE_LEGAL_KNOWLEDGE,
        "Legal",
        implemented=True,
        default_enabled=True,
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
