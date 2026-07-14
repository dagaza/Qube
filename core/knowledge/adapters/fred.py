"""FRED macroeconomic series search adapter (free API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import merge_query_params
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.FRED")

ADAPTER_ID = "fred"
RETRIEVAL_METHOD = "fred_series_search"
FRED_SERIES_SEARCH = "https://api.stlouisfed.org/fred/series/search"
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


def _row_from_series(series: dict[str, Any]) -> dict[str, Any] | None:
    series_id = str(series.get("id") or "").strip()
    title = str(series.get("title") or "").strip()
    notes = str(series.get("notes") or "").strip()
    if not series_id and not title:
        return None
    display_title = f"{series_id} — {title}" if series_id and title else (title or series_id)
    frequency = str(series.get("frequency_short") or series.get("frequency") or "").strip()
    units = str(series.get("units_short") or series.get("units") or "").strip()
    snippet_parts = [part for part in (notes, frequency, units) if part]
    snippet = ". ".join(snippet_parts)
    if not snippet:
        snippet = display_title
    snippet = snippet[:600]
    url = f"https://fred.stlouisfed.org/series/{series_id}" if series_id else None
    pub_date = str(series.get("last_updated") or series.get("observation_end") or "")[:10] or None
    return {
        "title": display_title,
        "snippet": snippet,
        "full_text": notes or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "FRED",
        "publication_date": pub_date,
        "document_type": "macro_series",
        "series_id": series_id or None,
        "frequency": frequency or None,
        "units": units or None,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("fred_search_unemployment.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[FRED] fixture load failed: %s", exc)

    from core.knowledge.credential_resolver import authorization_token

    if not authorization_token("fred"):
        logger.debug("[FRED] skipping live search (API key required)")
        return {"seriess": []}

    if not q:
        return {"seriess": []}

    try:
        resp = knowledge_get(
            FRED_SERIES_SEARCH,
            params=merge_query_params(
                {
                    "search_text": q,
                    "file_type": "json",
                    "limit": max(1, min(max_results, 10)),
                    "order_by": "search_rank",
                    "sort_order": "desc",
                },
                "fred",
            ),
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"seriess": []}
    except BudgetExhaustedError:
        logger.warning("[FRED] budget exhausted; skipping retry")
        return {"seriess": []}
    except Exception as exc:
        logger.warning("[FRED] search failed: %s", exc)
        return {"seriess": []}


def search_fred(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search FRED for macroeconomic data series (requires API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for series in payload.get("seriess") or []:
        if not isinstance(series, dict):
            continue
        row = _row_from_series(series)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
