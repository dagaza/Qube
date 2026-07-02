"""Europe PMC literature search adapter (open REST API, no key)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.EuropePMC")

ADAPTER_ID = "europe_pmc"
RETRIEVAL_METHOD = "europe_pmc_search"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"


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


def row_from_europe_pmc_entry(
    entry: dict[str, Any],
    *,
    adapter_id: str = ADAPTER_ID,
) -> dict[str, Any] | None:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstractText") or "").strip()
    if not title and not abstract:
        return None
    doi = str(entry.get("doi") or "").strip() or None
    pmid = str(entry.get("pmid") or "").strip()
    pmcid = str(entry.get("pmcid") or "").strip()
    url = f"https://doi.org/{doi}" if doi else None
    if not url and pmcid:
        url = f"https://europepmc.org/article/PMC/{pmcid.lstrip('PMC')}"
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
    source_type = str(entry.get("source") or "").upper()
    is_preprint = source_type == "PPR" or "preprint" in str(entry.get("pubType") or "").lower()
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": adapter_id,
        "authors": tuple(authors),
        "venue": str(entry.get("journalTitle") or "Europe PMC").strip(),
        "publication_date": pub_date or None,
        "doi": doi,
        "peer_reviewed": not is_preprint,
        "preprint": is_preprint,
        "open_access": bool(entry.get("isOpenAccess") in ("Y", True)),
        "document_type": "preprint" if is_preprint else "journal_abstract",
        "pmid": pmid or None,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    query_suffix: str = "",
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Query Europe PMC; optional ``query_suffix`` AND-clause (e.g. bioRxiv filter)."""
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("europe_pmc_search_trials.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Europe PMC] fixture load failed: %s", exc)

    if not q:
        return {"resultList": {"result": []}}

    query_expr = f"({q}){query_suffix}" if query_suffix else q
    try:
        resp = knowledge_get(
            EUROPE_PMC_SEARCH,
            params={
                "query": query_expr,
                "format": "json",
                "pageSize": max(1, min(max_results, 10)),
                "resultType": "core",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"resultList": {"result": []}}
    except BudgetExhaustedError:
        logger.warning("[Europe PMC] budget exhausted; skipping retry")
        return {"resultList": {"result": []}}
    except Exception as exc:
        logger.warning("[Europe PMC] search failed: %s", exc)
        return {"resultList": {"result": []}}


def search_europe_pmc(
    query: str,
    *,
    max_results: int = 3,
    query_suffix: str = "",
) -> list[dict[str, Any]]:
    """Search Europe PMC for scholarly abstracts and preprints."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(
        q,
        max_results=max_results,
        query_suffix=query_suffix,
    )
    result_list = payload.get("resultList") or {}
    entries = [
        entry for entry in (result_list.get("result") or []) if isinstance(entry, dict)
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = row_from_europe_pmc_entry(entry)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
