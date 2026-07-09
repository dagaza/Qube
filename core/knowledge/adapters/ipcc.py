"""IPCC adapter — assessment report discovery via IPCC-related Zenodo records."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.IPCC")

ADAPTER_ID = "ipcc"
RETRIEVAL_METHOD = "ipcc_zenodo_record_search"
ZENODO_URL = "https://zenodo.org/api/records"
USER_AGENT = "Qube/1.0 (local@qube.app)"
_TOKEN_SPLIT = re.compile(r"\s+")


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


def _zenodo_query(search_query: str) -> str:
    q = sanitize_api_query(search_query)
    if not q:
        return "IPCC"
    return f"IPCC {q}"


def _row_from_record(item: dict[str, Any]) -> dict[str, Any] | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    title = str(metadata.get("title") or "").strip()
    if not title:
        return None
    description = str(metadata.get("description") or "").strip()
    snippet = description[:600] if description else title
    doi = str(metadata.get("doi") or "").strip()
    record_id = str(item.get("id") or "").strip()
    links = item.get("links") if isinstance(item.get("links"), dict) else {}
    url = str(links.get("html") or links.get("self") or "").strip()
    if not url and doi:
        url = f"https://doi.org/{doi}"
    elif not url and record_id:
        url = f"https://zenodo.org/records/{record_id}"
    if not url:
        url = "https://www.ipcc.ch/report/"
    published = str(metadata.get("publication_date") or metadata.get("date") or "")[:10] or None
    creators = metadata.get("creators") or []
    author_names = tuple(
        str(c.get("name") or c).strip()
        for c in creators
        if isinstance(c, dict) and str(c.get("name") or "").strip()
    )[:5]
    if not author_names:
        author_names = ("Intergovernmental Panel on Climate Change",)
    return {
        "title": title,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": author_names,
        "venue": "IPCC Data Distribution Centre",
        "publication_date": published,
        "document_type": "assessment_report",
        "doi": doi or None,
        "zenodo_id": record_id or None,
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
        fixture = _fixture_search_path("ipcc_search_sea_level.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[IPCC] fixture load failed: %s", exc)

    if not q:
        return {"hits": {"hits": []}}

    try:
        resp = knowledge_get(
            ZENODO_URL,
            params={
                "q": _zenodo_query(q),
                "size": max(1, min(max_results, 10)),
                "sort": "bestmatch",
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"hits": {"hits": []}}
    except BudgetExhaustedError:
        logger.warning("[IPCC] budget exhausted; skipping retry")
        return {"hits": {"hits": []}}
    except Exception as exc:
        logger.warning("[IPCC] record search failed: %s", exc)
        return {"hits": {"hits": []}}


def search_ipcc(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search IPCC-related assessment records archived on Zenodo."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    hits = payload.get("hits") or {}
    records = hits.get("hits") if isinstance(hits, dict) else []
    rows: list[dict[str, Any]] = []
    for item in records or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_record(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
