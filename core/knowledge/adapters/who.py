"""WHO adapter — Global Health Observatory (GHO) indicator discovery."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.WHO")

ADAPTER_ID = "who"
RETRIEVAL_METHOD = "who_gho_indicator_search"
INDICATOR_URL = "https://ghoapi.azureedge.net/api/Indicator"
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


def _primary_term(search_query: str) -> str:
    terms = [
        term
        for term in _TOKEN_SPLIT.split(sanitize_api_query(search_query))
        if len(term) >= 3
    ]
    return terms[0] if terms else sanitize_api_query(search_query)


def _row_from_indicator(item: dict[str, Any]) -> dict[str, Any] | None:
    code = str(item.get("IndicatorCode") or item.get("indicator_code") or "").strip()
    name = str(item.get("IndicatorName") or item.get("indicator_name") or code or "").strip()
    if not name:
        return None
    display = f"{code} — {name}" if code else name
    url = (
        f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{code}"
        if code
        else "https://www.who.int/data/gho/"
    )
    return {
        "title": display,
        "snippet": name[:600],
        "full_text": name,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("World Health Organization",),
        "venue": "WHO Global Health Observatory",
        "publication_date": None,
        "document_type": "health_indicator",
        "indicator_code": code or None,
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
        fixture = _fixture_search_path("who_search_hypertension.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[WHO] fixture load failed: %s", exc)

    term = _primary_term(q)
    if not term:
        return {"value": []}

    safe_term = term.replace("'", "''")
    odata_filter = f"contains(IndicatorName,'{safe_term}')"
    try:
        resp = knowledge_get(
            INDICATOR_URL,
            params={
                "$filter": odata_filter,
                "$top": max(1, min(max_results, 10)),
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            values = payload.get("value") or []
            return {"value": values if isinstance(values, list) else []}
        return {"value": []}
    except BudgetExhaustedError:
        logger.warning("[WHO] budget exhausted; skipping retry")
        return {"value": []}
    except Exception as exc:
        logger.warning("[WHO] indicator search failed: %s", exc)
        return {"value": []}


def search_who(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search WHO GHO health indicators by keyword."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("value") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_indicator(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
