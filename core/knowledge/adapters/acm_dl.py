"""ACM Digital Library adapter — ACM-published works via OpenAlex publisher filter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.openalex import _reconstruct_abstract
from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import merge_query_params
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.ACMDL")

ADAPTER_ID = "acm_dl"
RETRIEVAL_METHOD = "acm_dl_search"
OPENALEX_WORKS = "https://api.openalex.org/works"
ACM_OPENALEX_PUBLISHER_ID = "P4310319798"
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


def _acm_url_from_work(work: dict[str, Any], doi: str | None) -> str | None:
    primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    landing = str(primary_location.get("landing_page_url") or "").strip()
    if landing and "dl.acm.org" in landing.lower():
        return landing
    if doi:
        return f"https://doi.org/{doi}"
    open_access = work.get("open_access") if isinstance(work.get("open_access"), dict) else {}
    oa_url = str(open_access.get("oa_url") or "").strip()
    if oa_url and "acm.org" in oa_url.lower():
        return oa_url
    return None


def _row_from_work(work: dict[str, Any]) -> dict[str, Any] | None:
    title = str(work.get("display_name") or work.get("title") or "").strip()
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    if not title and not abstract:
        return None
    doi_raw = work.get("doi")
    doi = str(doi_raw).replace("https://doi.org/", "").strip() if doi_raw else None
    url = _acm_url_from_work(work, doi)
    authors = tuple(
        str((a.get("author") or {}).get("display_name") or "").strip()
        for a in (work.get("authorships") or [])
        if isinstance(a, dict)
    )
    authors = tuple(a for a in authors if a)
    primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    venue = str(source.get("display_name") or "ACM Digital Library").strip()
    pub_year = work.get("publication_year")
    pub_date = str(pub_year) if pub_year else None
    oa = work.get("open_access") if isinstance(work.get("open_access"), dict) else {}
    open_access = bool(oa.get("is_oa")) if oa else None
    doc_type = str(work.get("type") or "").strip().lower()
    preprint = doc_type in {"preprint", "posted-content"}
    peer_reviewed = not preprint
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": venue,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": peer_reviewed,
        "preprint": preprint,
        "open_access": open_access,
        "document_type": "conference_paper" if "proceedings" in venue.lower() else "journal_abstract",
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _row_from_fixture(entry: dict[str, Any]) -> dict[str, Any] | None:
    title = str(entry.get("title") or "").strip()
    if not title:
        return None
    abstract = str(entry.get("abstract") or entry.get("snippet") or "").strip()
    authors_raw = entry.get("authors") or ()
    if isinstance(authors_raw, str):
        authors = (authors_raw.strip(),) if authors_raw.strip() else ()
    else:
        authors = tuple(str(a).strip() for a in authors_raw if str(a).strip())
    year = entry.get("year") or entry.get("publication_date")
    pub_date = str(year)[:4] if year else None
    return {
        "title": title,
        "snippet": (abstract or title)[:600],
        "full_text": abstract or None,
        "url": str(entry.get("url") or "").strip() or None,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": str(entry.get("venue") or "ACM Digital Library").strip(),
        "publication_date": pub_date,
        "doi": entry.get("doi"),
        "peer_reviewed": bool(entry.get("peer_reviewed", True)),
        "preprint": bool(entry.get("preprint", False)),
        "open_access": entry.get("open_access"),
        "document_type": str(entry.get("document_type") or "journal_abstract"),
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("acm_dl_search_transformer.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[ACM DL] fixture load failed: %s", exc)

    if not q:
        return {"results": []}

    try:
        resp = knowledge_get(
            OPENALEX_WORKS,
            params=merge_query_params(
                {
                    "search": q,
                    "filter": f"primary_location.source.host_organization:{ACM_OPENALEX_PUBLISHER_ID}",
                    "per_page": max(1, min(max_results, 10)),
                },
                "openalex",
            ),
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": []}
    except BudgetExhaustedError:
        logger.warning("[ACM DL] OpenAlex budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[ACM DL] OpenAlex ACM search failed: %s", exc)
        return {"results": []}


def search_acm_dl(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search ACM-published literature indexed in OpenAlex."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    if _use_fixtures() and isinstance(payload.get("results"), list):
        first = payload["results"][0] if payload["results"] else {}
        if isinstance(first, dict) and "abstract" in first and "display_name" not in first:
            for entry in payload.get("results") or []:
                if not isinstance(entry, dict):
                    continue
                row = _row_from_fixture(entry)
                if row is not None:
                    rows.append(row)
                if len(rows) >= max_results:
                    break
            return rows

    for work in payload.get("results") or []:
        if not isinstance(work, dict):
            continue
        row = _row_from_work(work)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
