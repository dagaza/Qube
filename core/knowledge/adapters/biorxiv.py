"""bioRxiv / life-science preprint adapter (fixture stub + Europe PMC live search)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.bioRxiv")

ADAPTER_ID = "biorxiv"
RETRIEVAL_METHOD = "biorxiv_search"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _fixture_search_path(name: str) -> Path | None:
    path = (
        Path(__file__).resolve().parents[3]
        / "eval"
        / "fixtures"
        / "knowledge"
        / name
    )
    return path if path.is_file() else None


def _use_fixtures() -> bool:
    return os.environ.get("QUBE_KNOWLEDGE_FIXTURES", "").strip() == "1"


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstract") or entry.get("snippet") or "").strip()
    url = str(entry.get("url") or "").strip() or None
    authors_raw = entry.get("authors") or ()
    if isinstance(authors_raw, str):
        authors = (authors_raw.strip(),) if authors_raw.strip() else ()
    else:
        authors = tuple(str(a).strip() for a in authors_raw if str(a).strip())
    year = entry.get("year") or entry.get("publication_date")
    pub_date = str(year)[:10] if year else None
    if pub_date and len(pub_date) == 4:
        pub_date = pub_date
    excerpt = abstract[:600] if abstract else title
    doi = entry.get("doi")
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": str(entry.get("venue") or "bioRxiv").strip(),
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": bool(entry.get("peer_reviewed", False)),
        "preprint": bool(entry.get("preprint", True)),
        "open_access": entry.get("open_access", True),
        "document_type": "preprint",
        "biorxiv_id": entry.get("biorxiv_id") or entry.get("id"),
    }


def _row_from_europe_pmc(entry: dict[str, Any]) -> dict[str, Any]:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstractText") or "").strip()
    doi = str(entry.get("doi") or "").strip() or None
    pmid = str(entry.get("pmid") or "").strip()
    url = f"https://doi.org/{doi}" if doi else None
    if not url and pmid:
        url = f"https://europepmc.org/article/MED/{pmid}"
    authors: list[str] = []
    author_list = entry.get("authorList") or {}
    if isinstance(author_list, dict):
        for author in author_list.get("author") or []:
            if isinstance(author, dict):
                name = str(author.get("fullName") or "").strip()
                if name:
                    authors.append(name)
    pub_date = str(entry.get("firstPublicationDate") or entry.get("pubYear") or "")[:10]
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": tuple(authors),
        "venue": str(entry.get("journalTitle") or "bioRxiv").strip(),
        "publication_date": pub_date or None,
        "doi": doi,
        "peer_reviewed": False,
        "preprint": True,
        "open_access": True,
        "document_type": "preprint",
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Load fixture rows or query Europe PMC for bioRxiv preprints (SRC:PPR)."""
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        for name in (
            "biorxiv_search_crispr.json",
            "biorxiv_search_microbiome.json",
        ):
            fixture = _fixture_search_path(name)
            if fixture is None:
                continue
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[bioRxiv] fixture load failed: %s", exc)
                continue
            fixture_query = str(payload.get("query") or "").lower()
            if fixture_query and fixture_query not in q.lower() and q.lower() not in fixture_query:
                continue
            return payload

    if not q:
        return {"results": []}

    try:
        resp = requests.get(
            EUROPE_PMC_SEARCH,
            params={
                "query": f"({q}) AND (SRC:PPR OR JOURNAL:\"bioRxiv\")",
                "format": "json",
                "pageSize": max(1, min(max_results, 10)),
                "resultType": "core",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("[bioRxiv] Europe PMC search failed: %s", exc)
        return {"results": []}

    results = []
    result_list = payload.get("resultList") or {}
    for entry in result_list.get("result") or []:
        if isinstance(entry, dict) and entry.get("title"):
            results.append(entry)
    return {"results": results, "source": "europe_pmc"}


def search_biorxiv(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search bioRxiv preprints via fixtures or Europe PMC preprint index."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    source = str(payload.get("source") or "fixture")
    entries = [
        entry for entry in (payload.get("results") or []) if isinstance(entry, dict)
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if source == "europe_pmc":
            row = _row_from_europe_pmc(entry)
        else:
            row = _row_from_entry(entry)
        if row.get("title"):
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
