"""BLS adapter — U.S. Bureau of Labor Statistics series search (API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_post

logger = logging.getLogger("Qube.Knowledge.BLS")

ADAPTER_ID = "bls"
RETRIEVAL_METHOD = "bls_series_search"
SERIES_SEARCH_URL = "https://api.bls.gov/publicAPI/v2/timeseries/search"
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
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _row_from_series(item: dict[str, Any]) -> dict[str, Any] | None:
    series_id = str(item.get("seriesID") or item.get("series_id") or "").strip()
    title = str(item.get("seriesTitle") or item.get("title") or series_id or "").strip()
    if not title:
        return None
    survey = str(item.get("surveyName") or item.get("survey_name") or "").strip()
    snippet_parts = [part for part in (survey, series_id) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else title
    url = (
        f"https://data.bls.gov/timeseries/{series_id}"
        if series_id
        else "https://www.bls.gov/data/"
    )
    display = f"{series_id} — {title}" if series_id else title
    return {
        "title": display,
        "snippet": snippet[:600],
        "full_text": title,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("U.S. Bureau of Labor Statistics",),
        "venue": "BLS",
        "publication_date": None,
        "document_type": "statistical_release",
        "series_id": series_id or None,
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
        fixture = _fixture_search_path("bls_search_unemployment.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[BLS] fixture load failed: %s", exc)

    api_key = authorization_token("bls")
    if not api_key:
        logger.debug("[BLS] skipping live search (API key required)")
        return {"series": []}

    if not q:
        return {"series": []}

    body = {
        "series_text": q,
        "registrationkey": api_key,
        "limit": max(1, min(max_results, 10)),
    }
    try:
        resp = knowledge_post(
            SERIES_SEARCH_URL,
            json=body,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            return {"series": []}
        results = payload.get("Results") or payload.get("results") or payload
        if isinstance(results, dict):
            series = results.get("series") or results.get("Series") or []
            return {"series": series if isinstance(series, list) else []}
        return {"series": []}
    except BudgetExhaustedError:
        logger.warning("[BLS] budget exhausted; skipping retry")
        return {"series": []}
    except Exception as exc:
        logger.warning("[BLS] series search failed: %s", exc)
        return {"series": []}


def search_bls(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search BLS time series catalog (requires free registration key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("series") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_series(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
