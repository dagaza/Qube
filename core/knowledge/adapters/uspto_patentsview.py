"""USPTO PatentsView PatentSearch adapter."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_post

logger = logging.getLogger("Qube.Knowledge.PatentsView")

ADAPTER_ID = "uspto_patentsview"
RETRIEVAL_METHOD = "patentsview_patent_search"
PATENTSEARCH_URL = "https://search.patentsview.org/api/v1/patent"
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


def _api_key() -> str | None:
    return authorization_token("patentsview")


def _headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["X-Api-Key"] = key
    return headers


def _search_query(query: str) -> dict[str, Any]:
    tokens = [token for token in re.split(r"\s+", query) if len(token) >= 2]
    if not tokens:
        tokens = [query]
    if len(tokens) == 1:
        return {"patent_title": {"_text_any": tokens[0]}}
    return {"patent_title": {"_text_all": tokens}}


def _patent_url(patent_id: str | None, patent_number: str | None) -> str | None:
    number = str(patent_number or patent_id or "").strip()
    if not number:
        return None
    digits = re.sub(r"\D", "", number)
    if not digits:
        return None
    return f"https://patents.google.com/patent/US{digits}"


def _row_from_patent(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("patent_title") or "").strip()
    patent_id = str(item.get("patent_id") or item.get("patent_number") or "").strip()
    if not title and not patent_id:
        return None
    patent_date = str(item.get("patent_date") or "")[:10] or None
    assignees = item.get("assignees") or item.get("assignee_organization")
    author = "USPTO PatentsView"
    if isinstance(assignees, list) and assignees:
        first = assignees[0]
        if isinstance(first, dict):
            author = str(first.get("assignee_organization") or author).strip() or author
        else:
            author = str(first).strip() or author
    elif isinstance(assignees, str) and assignees.strip():
        author = assignees.strip()
    snippet = title
    if patent_date:
        snippet = f"{title} (granted {patent_date})"
    return {
        "title": title or f"U.S. Patent {patent_id}",
        "snippet": snippet[:600],
        "full_text": None,
        "url": _patent_url(patent_id, str(item.get("patent_number") or "")),
        "_adapter": ADAPTER_ID,
        "authors": (author,),
        "venue": "USPTO PatentsView",
        "publication_date": patent_date,
        "document_type": "patent",
        "patent_id": patent_id or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("uspto_patentsview_search_battery.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[PatentsView] fixture load failed: %s", exc)

    if not q or not _api_key():
        return {"patents": []}

    payload = {
        "q": _search_query(q),
        "f": ["patent_id", "patent_title", "patent_date", "assignee_organization"],
        "o": {"per_page": max(1, min(max_results, 10))},
    }
    try:
        resp = knowledge_post(
            PATENTSEARCH_URL,
            json=payload,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, dict) else {"patents": []}
    except BudgetExhaustedError:
        logger.warning("[PatentsView] budget exhausted; skipping retry")
        return {"patents": []}
    except Exception as exc:
        logger.warning("[PatentsView] patent search failed: %s", exc)
        return {"patents": []}


def search_uspto_patentsview(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search U.S. granted patents via PatentsView PatentSearch API."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("patents") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_patent(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
