"""Eurostat adapter — official EU statistics discovery API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.Eurostat")

ADAPTER_ID = "eurostat"
RETRIEVAL_METHOD = "eurostat_statistics_search"
SEARCH_URL = "https://ec.europa.eu/eurostat/api/discovery/statistics"
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


def _row_from_stat(item: dict[str, Any]) -> dict[str, Any] | None:
    code = str(item.get("code") or item.get("id") or "").strip()
    title = str(item.get("title") or item.get("label") or code or "").strip()
    if not title:
        return None
    description = str(item.get("description") or item.get("summary") or "").strip()
    snippet = description[:600] if description else title
    url = (
        f"https://ec.europa.eu/eurostat/databrowser/view/{code}/default/table"
        if code
        else "https://ec.europa.eu/eurostat"
    )
    display = f"{code} — {title}" if code else title
    return {
        "title": display,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Eurostat",),
        "venue": "Eurostat",
        "publication_date": None,
        "document_type": "statistical_release",
        "dataset_code": code or None,
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
        fixture = _fixture_search_path("eurostat_search_unemployment.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Eurostat] fixture load failed: %s", exc)

    if not q:
        return {"statistics": []}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={"query": q, "format": "json", "lang": "en", "size": max(1, min(max_results, 10))},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            stats = payload.get("statistics") or payload.get("results") or payload.get("items") or []
            return {"statistics": stats if isinstance(stats, list) else []}
        return {"statistics": []}
    except BudgetExhaustedError:
        logger.warning("[Eurostat] budget exhausted; skipping retry")
        return {"statistics": []}
    except Exception as exc:
        logger.warning("[Eurostat] search failed: %s", exc)
        return {"statistics": []}


def search_eurostat(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search Eurostat official statistics datasets."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("statistics") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_stat(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
