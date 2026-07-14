"""USGS adapter — publication search via USGS Publications Service."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.USGS")

ADAPTER_ID = "usgs"
RETRIEVAL_METHOD = "usgs_publication_search"
PUBLICATIONS_URL = "https://pubs.usgs.gov/pubs-services/publication/search"
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


def _row_from_publication(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    pub_id = item.get("id") or item.get("publicationId")
    summary = str(item.get("abstract") or item.get("summary") or "").strip()
    snippet = summary[:600] if summary else title
    pub_year = str(item.get("publicationYear") or item.get("year") or "")[:4] or None
    url = str(item.get("publicationUrl") or item.get("url") or "").strip()
    if not url and pub_id is not None:
        url = f"https://pubs.usgs.gov/publication/{pub_id}"
    if not url:
        url = "https://www.usgs.gov/products/publications"
    return {
        "title": title,
        "snippet": snippet,
        "full_text": summary or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": tuple(
            str(a).strip()
            for a in (item.get("authors") or item.get("author") or ())
            if str(a).strip()
        )[:5],
        "venue": "USGS Publications",
        "publication_date": pub_year,
        "document_type": "government_publication",
        "publication_id": str(pub_id) if pub_id is not None else None,
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
        fixture = _fixture_search_path("usgs_search_earthquake.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[USGS] fixture load failed: %s", exc)

    if not q:
        return {"records": []}

    try:
        resp = knowledge_get(
            PUBLICATIONS_URL,
            params={"title": q, "max": max(1, min(max_results, 10))},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            return {"records": payload}
        if isinstance(payload, dict):
            records = payload.get("records") or payload.get("publications") or []
            return {"records": records if isinstance(records, list) else []}
        return {"records": []}
    except BudgetExhaustedError:
        logger.warning("[USGS] budget exhausted; skipping retry")
        return {"records": []}
    except Exception as exc:
        logger.warning("[USGS] publication search failed: %s", exc)
        return {"records": []}


def search_usgs(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search USGS publications for geoscience and environmental reports."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("records") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_publication(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
