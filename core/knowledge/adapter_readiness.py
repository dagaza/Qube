"""Canonical readiness and production-strategy metadata per knowledge adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.knowledge.adapters.registry import SEARCH_FUNCTIONS

ReadinessLevel = Literal["stub", "beta", "production"]


@dataclass(frozen=True)
class AdapterReadinessMeta:
    readiness: ReadinessLevel
    production_strategy: str


# Source of truth for production strategy descriptions (also drives readiness when
# ``implemented=True`` unless overridden below).
_ADAPTER_STRATEGY: dict[str, AdapterReadinessMeta] = {
    "pubmed": AdapterReadinessMeta(
        "production",
        "NCBI E-utilities direct search (optional NCBI API key).",
    ),
    "openalex": AdapterReadinessMeta(
        "production",
        "OpenAlex REST works search (optional free API key).",
    ),
    "crossref": AdapterReadinessMeta(
        "production",
        "Crossref REST works metadata (polite pool, anonymous).",
    ),
    "semantic_scholar": AdapterReadinessMeta(
        "beta",
        "Semantic Scholar Graph API (free API key required).",
    ),
    "europe_pmc": AdapterReadinessMeta(
        "production",
        "Europe PMC REST search (anonymous).",
    ),
    "arxiv": AdapterReadinessMeta(
        "production",
        "arXiv Atom API (anonymous).",
    ),
    "biorxiv": AdapterReadinessMeta(
        "production",
        "bioRxiv preprints via Europe PMC filter (anonymous).",
    ),
    "inspire_hep": AdapterReadinessMeta(
        "production",
        "INSPIRE-HEP REST literature search (anonymous).",
    ),
    "nasa_ads": AdapterReadinessMeta(
        "beta",
        "NASA ADS REST search (personal API token required).",
    ),
    "socarxiv": AdapterReadinessMeta(
        "production",
        "SocArXiv preprints via OSF API (anonymous).",
    ),
    "ssrn": AdapterReadinessMeta(
        "beta",
        "SSRN works via OpenAlex source filter (optional OpenAlex key).",
    ),
    "psyarxiv": AdapterReadinessMeta(
        "beta",
        "PsyArXiv preprints via OSF API (anonymous, opt-in).",
    ),
    "noaa": AdapterReadinessMeta(
        "beta",
        "NOAA NCEI CDO datasets API (token required).",
    ),
    "nasa_earthdata": AdapterReadinessMeta(
        "beta",
        "NASA Earthdata CMR collections JSON search (anonymous, opt-in).",
    ),
    "pubchem": AdapterReadinessMeta(
        "production",
        "PubChem PUG REST compound search (optional NCBI key).",
    ),
    "dblp": AdapterReadinessMeta(
        "production",
        "DBLP publication search API (anonymous).",
    ),
    "acm_dl": AdapterReadinessMeta(
        "beta",
        "ACM works via OpenAlex publisher filter (optional OpenAlex key).",
    ),
    "repec": AdapterReadinessMeta(
        "production",
        "RePEc/IDEAS metadata via EconBiz API (anonymous).",
    ),
    "psycinfo": AdapterReadinessMeta(
        "beta",
        "PsycINFO via institutional EBSCO EDS API (credentials required).",
    ),
    "sec_edgar": AdapterReadinessMeta(
        "production",
        "SEC EDGAR submissions JSON (anonymous).",
    ),
    "fred": AdapterReadinessMeta(
        "production",
        "FRED series search API (free API key required).",
    ),
    "companies_house": AdapterReadinessMeta(
        "beta",
        "UK Companies House REST search (free API key required).",
    ),
    "alpha_vantage": AdapterReadinessMeta(
        "beta",
        "Alpha Vantage SYMBOL_SEARCH (free API key required).",
    ),
    "bloomberg_api": AdapterReadinessMeta(
        "beta",
        "Bloomberg Open API via local HTTP bridge (enterprise URL required).",
    ),
    "courtlistener": AdapterReadinessMeta(
        "production",
        "CourtListener v4 REST search (optional free account token).",
    ),
    "eur_lex": AdapterReadinessMeta(
        "beta",
        "EUR-Lex CELLAR SPARQL legal-act search (anonymous, opt-in).",
    ),
    "canlii": AdapterReadinessMeta(
        "beta",
        "CanLII REST case search (free API key required).",
    ),
    "bailii": AdapterReadinessMeta(
        "beta",
        "BAILII HTML search (no official API; respectful scrape).",
    ),
    "clinicaltrials_gov": AdapterReadinessMeta(
        "production",
        "ClinicalTrials.gov REST API v2 study search (anonymous).",
    ),
    "openfda": AdapterReadinessMeta(
        "production",
        "openFDA drug label search API (anonymous).",
    ),
    "world_bank": AdapterReadinessMeta(
        "production",
        "World Bank Open Data indicator catalog search (anonymous).",
    ),
    "eurostat": AdapterReadinessMeta(
        "production",
        "Eurostat discovery statistics search API (anonymous).",
    ),
    "usgs": AdapterReadinessMeta(
        "production",
        "USGS Publications Service search (anonymous).",
    ),
    "usda_fdc": AdapterReadinessMeta(
        "production",
        "USDA FoodData Central REST search (optional free API key).",
    ),
    "nist": AdapterReadinessMeta(
        "production",
        "NIST NVD keyword search (optional free API key).",
    ),
    "ietf_rfc": AdapterReadinessMeta(
        "production",
        "IETF Datatracker document search (anonymous).",
    ),
    "bls": AdapterReadinessMeta(
        "production",
        "BLS Public Data API series search (free registration key required).",
    ),
    "us_census": AdapterReadinessMeta(
        "production",
        "U.S. Census Bureau data.json catalog search (optional free API key).",
    ),
    "ieee_xplore": AdapterReadinessMeta(
        "beta",
        "IEEE Xplore Metadata API (free developer key required).",
    ),
    "oecd": AdapterReadinessMeta(
        "production",
        "OECD SDMX dataflow catalog keyword search (anonymous).",
    ),
    "nice": AdapterReadinessMeta(
        "beta",
        "NICE syndication guidance index search (syndication API key required).",
    ),
    "cdc": AdapterReadinessMeta(
        "production",
        "CDC Content Services media search and Open Data catalog (anonymous).",
    ),
    "who": AdapterReadinessMeta(
        "production",
        "WHO Global Health Observatory indicator search (anonymous OData).",
    ),
    "ipcc": AdapterReadinessMeta(
        "production",
        "IPCC-related assessment records via Zenodo discovery search (anonymous).",
    ),
    "fao": AdapterReadinessMeta(
        "beta",
        "FAOSTAT dataset catalog search (FAOSTAT API bearer token required).",
    ),
    "usda": AdapterReadinessMeta(
        "production",
        "USDA ERS ARMS variable search (optional api.data.gov key).",
    ),
    "copernicus_cds": AdapterReadinessMeta(
        "production",
        "Copernicus CDS STAC catalogue search (anonymous; CDS token for downloads).",
    ),
    "openreview": AdapterReadinessMeta(
        "production",
        "OpenReview notes search API (anonymous).",
    ),
    "acl_anthology": AdapterReadinessMeta(
        "beta",
        "ACL Anthology metadata via Verbatim search API (anonymous; no official REST search).",
    ),
    "chembl": AdapterReadinessMeta(
        "production",
        "ChEMBL molecule search REST API (anonymous).",
    ),
    "uniprot": AdapterReadinessMeta(
        "production",
        "UniProtKB REST search API (anonymous).",
    ),
    "pdb": AdapterReadinessMeta(
        "production",
        "RCSB PDB Search API + core entry metadata (anonymous).",
    ),
    "chemrxiv": AdapterReadinessMeta(
        "production",
        "ChemRxiv preprints via Europe PMC DOI prefix filter (anonymous).",
    ),
    "congress_gov": AdapterReadinessMeta(
        "beta",
        "Congress.gov bill metadata search (api.data.gov key required).",
    ),
    "govinfo": AdapterReadinessMeta(
        "beta",
        "GovInfo federal publication search API (api.data.gov key required).",
    ),
    "legislation_uk": AdapterReadinessMeta(
        "production",
        "UK legislation title search via legislation.gov.uk Atom feed (anonymous).",
    ),
    "uspto_patentsview": AdapterReadinessMeta(
        "beta",
        "USPTO PatentsView PatentSearch API (PatentsView API key required).",
    ),
    "epo_espacenet": AdapterReadinessMeta(
        "beta",
        "EPO Open Patent Services bibliographic search (consumer key + secret required).",
    ),
}


def _derive_readiness(
    *,
    implemented: bool,
    requires_api_key: bool,
    default_enabled: bool,
) -> ReadinessLevel:
    if not implemented:
        return "stub"
    if requires_api_key or not default_enabled:
        return "beta"
    return "production"


def get_adapter_readiness_meta(
    adapter_id: str,
    *,
    implemented: bool | None = None,
    requires_api_key: bool = False,
    default_enabled: bool = True,
    explicit_readiness: ReadinessLevel | None = None,
    explicit_strategy: str = "",
) -> AdapterReadinessMeta:
    """Resolve readiness metadata for one adapter id."""
    aid = (adapter_id or "").strip().lower()
    if not aid:
        return AdapterReadinessMeta("stub", "Unknown adapter.")

    in_registry = aid in SEARCH_FUNCTIONS
    if implemented is None:
        implemented = in_registry

    canonical = _ADAPTER_STRATEGY.get(aid)
    if not implemented or not in_registry:
        strategy = explicit_strategy or (canonical.production_strategy if canonical else "")
        if not strategy:
            strategy = "Catalog placeholder; not registered in SEARCH_FUNCTIONS."
        return AdapterReadinessMeta(
            explicit_readiness or "stub",
            strategy,
        )

    readiness = (
        explicit_readiness
        or (canonical.readiness if canonical else None)
        or _derive_readiness(
            implemented=True,
            requires_api_key=requires_api_key,
            default_enabled=default_enabled,
        )
    )
    strategy = explicit_strategy or (canonical.production_strategy if canonical else "Live adapter.")
    return AdapterReadinessMeta(readiness, strategy)


def readiness_for_catalog_entry(entry) -> AdapterReadinessMeta:
    """Convenience wrapper for ``AdapterCatalogEntry`` instances."""
    return get_adapter_readiness_meta(
        entry.id,
        implemented=entry.implemented,
        requires_api_key=entry.requires_api_key,
        default_enabled=entry.default_enabled,
        explicit_readiness=entry.readiness,
        explicit_strategy=entry.production_strategy,
    )


def implemented_adapter_readiness() -> dict[str, AdapterReadinessMeta]:
    """One row per live adapter id (registry keys)."""
    from core.knowledge.adapters.catalog import get_adapter_entry

    out: dict[str, AdapterReadinessMeta] = {}
    for adapter_id in sorted(SEARCH_FUNCTIONS):
        entry = get_adapter_entry(adapter_id)
        if entry is not None:
            out[adapter_id] = readiness_for_catalog_entry(entry)
        else:
            out[adapter_id] = get_adapter_readiness_meta(adapter_id)
    return out


def adapters_by_readiness(level: ReadinessLevel) -> tuple[str, ...]:
    return tuple(
        aid
        for aid, meta in implemented_adapter_readiness().items()
        if meta.readiness == level
    )
