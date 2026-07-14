"""NASA ADS adapter — astrophysics literature search (API token required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.NASA_ADS")

ADAPTER_ID = "nasa_ads"
RETRIEVAL_METHOD = "ads_search"
SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
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
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = authorization_token("nasa_ads")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _row_from_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
    title_parts = doc.get("title") or []
    title = str(title_parts[0] if title_parts else "").strip()
    abstract = str(doc.get("abstract") or "").strip()
    if not title and not abstract:
        return None
    authors = tuple(str(a).strip() for a in (doc.get("author") or []) if str(a).strip())
    bibcode = str(doc.get("bibcode") or "").strip()
    doi_list = doc.get("doi") or []
    doi = str(doi_list[0]).strip().lower() if doi_list else None
    url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}" if bibcode else None
    if not url and doi:
        url = f"https://doi.org/{doi}"
    pub_date = str(doc.get("pubdate") or doc.get("year") or "")[:4] or None
    venue_parts = doc.get("pub") or doc.get("bibstem") or []
    venue = str(venue_parts[0] if isinstance(venue_parts, list) and venue_parts else venue_parts or "").strip()
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": venue or None,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": True,
        "preprint": bool(doc.get("preprint")),
        "open_access": None,
        "document_type": "journal_abstract",
        "bibcode": bibcode or None,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if not q:
        return {"response": {"docs": []}}

    if _use_fixtures():
        fixture = _fixture_search_path("nasa_ads_search_ligo.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[NASA ADS] fixture load failed: %s", exc)

    if not authorization_token("nasa_ads"):
        logger.debug("[NASA ADS] skipping live search (API token required)")
        return {"response": {"docs": []}}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={
                "q": q,
                "rows": max(1, min(max_results, 10)),
                "fl": "title,abstract,author,bibcode,doi,pubdate,year,pub,bibstem,preprint",
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"response": {"docs": []}}
    except BudgetExhaustedError:
        logger.warning("[NASA ADS] budget exhausted; skipping retry")
        return {"response": {"docs": []}}
    except Exception as exc:
        logger.warning("[NASA ADS] search failed: %s", exc)
        return {"response": {"docs": []}}


def search_nasa_ads(
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Search NASA ADS (requires configured API token)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results, timeout=timeout)
    response = payload.get("response") if isinstance(payload, dict) else {}
    docs = response.get("docs") if isinstance(response, dict) else []
    rows: list[dict[str, Any]] = []
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        row = _row_from_doc(doc)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
