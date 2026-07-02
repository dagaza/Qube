"""Semantic Scholar paper search adapter (API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.SemanticScholar")

ADAPTER_ID = "semantic_scholar"
RETRIEVAL_METHOD = "semantic_scholar_search"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
USER_AGENT = "Qube/1.0 (local@qube.app)"
_FIELDS = "paperId,title,abstract,authors,year,externalIds,url,venue,openAccessPdf,isOpenAccess"


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


def _headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = authorization_token("semantic_scholar")
    if token:
        headers["x-api-key"] = token
    return headers


def _row_from_paper(paper: dict[str, Any]) -> dict[str, Any] | None:
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    if not title and not abstract:
        return None
    authors: list[str] = []
    for author in paper.get("authors") or []:
        if not isinstance(author, dict):
            continue
        name = str(author.get("name") or "").strip()
        if name:
            authors.append(name)
    external = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
    doi_raw = external.get("DOI") if isinstance(external, dict) else None
    doi = str(doi_raw).strip().lower() if doi_raw else None
    url = str(paper.get("url") or "").strip() or None
    if not url and doi:
        url = f"https://doi.org/{doi}"
    open_pdf = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else {}
    if not url and isinstance(open_pdf, dict):
        url = str(open_pdf.get("url") or "").strip() or None
    year = paper.get("year")
    pub_date = str(year) if year else None
    venue = str(paper.get("venue") or "").strip()
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": tuple(authors),
        "venue": venue or None,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": bool(paper.get("isOpenAccess")),
        "document_type": "journal_abstract",
        "paper_id": paper.get("paperId"),
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if not q:
        return {"data": []}

    if _use_fixtures():
        fixture = _fixture_search_path("semantic_scholar_search_transformer.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Semantic Scholar] fixture load failed: %s", exc)

    if not authorization_token("semantic_scholar"):
        logger.debug("[Semantic Scholar] skipping live search (API key required)")
        return {"data": []}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={
                "query": q,
                "limit": max(1, min(max_results, 10)),
                "fields": _FIELDS,
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"data": []}
    except BudgetExhaustedError:
        logger.warning("[Semantic Scholar] budget exhausted; skipping retry")
        return {"data": []}
    except Exception as exc:
        logger.warning("[Semantic Scholar] search failed: %s", exc)
        return {"data": []}


def search_semantic_scholar(
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar (requires configured API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results, timeout=timeout)
    papers = payload.get("data") if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for paper in papers or []:
        if not isinstance(paper, dict):
            continue
        row = _row_from_paper(paper)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
