"""IEEE Xplore adapter — engineering literature search (API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import merge_query_params
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.IEEE")

ADAPTER_ID = "ieee_xplore"
RETRIEVAL_METHOD = "ieee_xplore_article_search"
SEARCH_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
USER_AGENT = "Qube/1.0 (local@qube.app)"


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
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _row_from_article(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(item.get("title") or item.get("article_title") or "").strip()
    if not title:
        return None
    abstract = str(item.get("abstract") or "").strip()
    snippet = abstract[:600] if abstract else title
    url = str(item.get("html_url") or item.get("pdf_url") or item.get("documentLink") or "").strip()
    if not url:
        article_number = str(item.get("article_number") or item.get("articleNumber") or "").strip()
        if article_number:
            url = f"https://ieeexplore.ieee.org/document/{article_number}"
        else:
            url = "https://ieeexplore.ieee.org/"
    authors_raw = item.get("authors") or item.get("author") or []
    authors: list[str] = []
    if isinstance(authors_raw, dict):
        authors_raw = authors_raw.get("authors") or []
    if isinstance(authors_raw, list):
        for author in authors_raw:
            if isinstance(author, dict):
                name = str(author.get("full_name") or author.get("name") or "").strip()
                if name:
                    authors.append(name)
            elif str(author).strip():
                authors.append(str(author).strip())
    pub_year = str(item.get("publication_year") or item.get("publicationYear") or "")[:4] or None
    return {
        "title": title,
        "snippet": snippet,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": tuple(authors[:5]),
        "venue": str(item.get("publication_title") or "IEEE Xplore"),
        "publication_date": pub_year,
        "document_type": "journal_abstract",
        "doi": str(item.get("doi") or "").strip() or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 15.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("ieee_xplore_search_robotics.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[IEEE] fixture load failed: %s", exc)

    params = merge_query_params(
        {
            "querytext": q,
            "max_records": max(1, min(max_results, 10)),
        },
        "ieee_xplore",
    )
    from core.knowledge.credential_resolver import authorization_token

    api_key = authorization_token("ieee_xplore") or params.pop("api_key", None)
    if api_key:
        params["apikey"] = api_key
    if not params.get("apikey"):
        logger.debug("[IEEE] skipping live search (API key required)")
        return {"articles": []}

    if not q:
        return {"articles": []}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params=params,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            articles = payload.get("articles") or payload.get("records") or []
            return {"articles": articles if isinstance(articles, list) else []}
        return {"articles": []}
    except BudgetExhaustedError:
        logger.warning("[IEEE] budget exhausted; skipping retry")
        return {"articles": []}
    except Exception as exc:
        logger.warning("[IEEE] search failed: %s", exc)
        return {"articles": []}


def search_ieee_xplore(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search IEEE Xplore for engineering literature (requires API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("articles") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_article(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
