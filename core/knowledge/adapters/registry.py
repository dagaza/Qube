"""Dispatch table for implemented knowledge adapters."""

from __future__ import annotations

from typing import Any, Callable

from core.knowledge.adapters.acm_dl import search_acm_dl
from core.knowledge.adapters.acl_anthology import search_acl_anthology
from core.knowledge.adapters.chembl import search_chembl
from core.knowledge.adapters.chemrxiv import search_chemrxiv
from core.knowledge.adapters.alpha_vantage import search_alpha_vantage
from core.knowledge.adapters.bloomberg_api import search_bloomberg_api
from core.knowledge.adapters.bls import search_bls
from core.knowledge.adapters.cdc import search_cdc
from core.knowledge.adapters.copernicus_cds import search_copernicus_cds
from core.knowledge.adapters.fao import search_fao
from core.knowledge.adapters.ipcc import search_ipcc
from core.knowledge.adapters.clinicaltrials_gov import search_clinicaltrials_gov
from core.knowledge.adapters.eurostat import search_eurostat
from core.knowledge.adapters.ieee_xplore import search_ieee_xplore
from core.knowledge.adapters.ietf_rfc import search_ietf_rfc
from core.knowledge.adapters.legislation_uk import search_legislation_uk
from core.knowledge.adapters.nice import search_nice
from core.knowledge.adapters.nist import search_nist
from core.knowledge.adapters.oecd import search_oecd
from core.knowledge.adapters.openfda import search_openfda
from core.knowledge.adapters.bailii import search_bailii
from core.knowledge.adapters.canlii import search_canlii
from core.knowledge.adapters.companies_house import search_companies_house
from core.knowledge.adapters.congress_gov import search_congress_gov
from core.knowledge.adapters.courtlistener import search_courtlistener
from core.knowledge.adapters.eur_lex import search_eur_lex
from core.knowledge.adapters.arxiv_api import search_arxiv
from core.knowledge.adapters.biorxiv import search_biorxiv
from core.knowledge.adapters.crossref import search_crossref
from core.knowledge.adapters.dblp import search_dblp
from core.knowledge.adapters.europe_pmc import search_europe_pmc
from core.knowledge.adapters.epo_espacenet import search_epo_espacenet
from core.knowledge.adapters.fred import search_fred
from core.knowledge.adapters.govinfo import search_govinfo
from core.knowledge.adapters.inspire_hep import search_inspire_hep
from core.knowledge.adapters.nasa_ads import search_nasa_ads
from core.knowledge.adapters.openalex import search_openalex
from core.knowledge.adapters.openreview import search_openreview
from core.knowledge.adapters.pdb import search_pdb
from core.knowledge.adapters.pubchem import search_pubchem
from core.knowledge.adapters.pubmed_eutils import search_pubmed
from core.knowledge.adapters.nasa_earthdata import search_nasa_earthdata
from core.knowledge.adapters.noaa import search_noaa
from core.knowledge.adapters.psyarxiv import search_psyarxiv
from core.knowledge.adapters.psycinfo import search_psycinfo
from core.knowledge.adapters.repec import search_repec
from core.knowledge.adapters.sec_edgar import search_sec_edgar
from core.knowledge.adapters.semantic_scholar import search_semantic_scholar
from core.knowledge.adapters.socarxiv import search_socarxiv
from core.knowledge.adapters.ssrn import search_ssrn
from core.knowledge.adapters.us_census import search_us_census
from core.knowledge.adapters.usda import search_usda
from core.knowledge.adapters.usda_fdc import search_usda_fdc
from core.knowledge.adapters.usgs import search_usgs
from core.knowledge.adapters.uniprot import search_uniprot
from core.knowledge.adapters.uspto_patentsview import search_uspto_patentsview
from core.knowledge.adapters.who import search_who
from core.knowledge.adapters.world_bank import search_world_bank

SearchFn = Callable[..., list[dict[str, Any]]]

SEARCH_FUNCTIONS: dict[str, SearchFn] = {
    "pubmed": search_pubmed,
    "openalex": search_openalex,
    "crossref": search_crossref,
    "semantic_scholar": search_semantic_scholar,
    "europe_pmc": search_europe_pmc,
    "arxiv": search_arxiv,
    "biorxiv": search_biorxiv,
    "inspire_hep": search_inspire_hep,
    "nasa_ads": search_nasa_ads,
    "socarxiv": search_socarxiv,
    "ssrn": search_ssrn,
    "psyarxiv": search_psyarxiv,
    "noaa": search_noaa,
    "nasa_earthdata": search_nasa_earthdata,
    "pubchem": search_pubchem,
    "dblp": search_dblp,
    "acm_dl": search_acm_dl,
    "repec": search_repec,
    "psycinfo": search_psycinfo,
    "sec_edgar": search_sec_edgar,
    "fred": search_fred,
    "companies_house": search_companies_house,
    "alpha_vantage": search_alpha_vantage,
    "bloomberg_api": search_bloomberg_api,
    "courtlistener": search_courtlistener,
    "eur_lex": search_eur_lex,
    "canlii": search_canlii,
    "bailii": search_bailii,
    "clinicaltrials_gov": search_clinicaltrials_gov,
    "openfda": search_openfda,
    "world_bank": search_world_bank,
    "eurostat": search_eurostat,
    "usgs": search_usgs,
    "usda_fdc": search_usda_fdc,
    "nist": search_nist,
    "ietf_rfc": search_ietf_rfc,
    "bls": search_bls,
    "us_census": search_us_census,
    "ieee_xplore": search_ieee_xplore,
    "oecd": search_oecd,
    "nice": search_nice,
    "cdc": search_cdc,
    "who": search_who,
    "ipcc": search_ipcc,
    "fao": search_fao,
    "usda": search_usda,
    "copernicus_cds": search_copernicus_cds,
    "openreview": search_openreview,
    "acl_anthology": search_acl_anthology,
    "chembl": search_chembl,
    "uniprot": search_uniprot,
    "pdb": search_pdb,
    "chemrxiv": search_chemrxiv,
    "congress_gov": search_congress_gov,
    "govinfo": search_govinfo,
    "legislation_uk": search_legislation_uk,
    "uspto_patentsview": search_uspto_patentsview,
    "epo_espacenet": search_epo_espacenet,
}


def get_search_function(adapter_id: str) -> SearchFn | None:
    return SEARCH_FUNCTIONS.get((adapter_id or "").strip().lower())
