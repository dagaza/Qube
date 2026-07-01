"""Dispatch table for implemented knowledge adapters."""

from __future__ import annotations

from typing import Any, Callable

from core.knowledge.adapters.courtlistener import search_courtlistener
from core.knowledge.adapters.arxiv_api import search_arxiv
from core.knowledge.adapters.biorxiv import search_biorxiv
from core.knowledge.adapters.dblp import search_dblp
from core.knowledge.adapters.inspire_hep import search_inspire_hep
from core.knowledge.adapters.openalex import search_openalex
from core.knowledge.adapters.pubchem import search_pubchem
from core.knowledge.adapters.pubmed_eutils import search_pubmed
from core.knowledge.adapters.repec import search_repec
from core.knowledge.adapters.sec_edgar import search_sec_edgar

SearchFn = Callable[..., list[dict[str, Any]]]

SEARCH_FUNCTIONS: dict[str, SearchFn] = {
    "pubmed": search_pubmed,
    "openalex": search_openalex,
    "arxiv": search_arxiv,
    "biorxiv": search_biorxiv,
    "inspire_hep": search_inspire_hep,
    "pubchem": search_pubchem,
    "dblp": search_dblp,
    "repec": search_repec,
    "sec_edgar": search_sec_edgar,
    "courtlistener": search_courtlistener,
}


def get_search_function(adapter_id: str) -> SearchFn | None:
    return SEARCH_FUNCTIONS.get((adapter_id or "").strip().lower())
