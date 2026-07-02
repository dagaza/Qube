"""World Bank Open Data adapter — macro indicator discovery."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.WorldBank")

ADAPTER_ID = "world_bank"
RETRIEVAL_METHOD = "world_bank_indicator_search"
INDICATORS_URL = "https://api.worldbank.org/v2/indicator"
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


def _query_terms(search_query: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in _TOKEN_SPLIT.split(sanitize_api_query(search_query).lower())
        if len(term) >= 3
    )


def _indicator_score(indicator: dict[str, Any], terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    haystack = " ".join(
        str(indicator.get(key) or "")
        for key in ("id", "name", "sourceNote", "sourceOrganization")
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _row_from_indicator(indicator: dict[str, Any]) -> dict[str, Any] | None:
    indicator_id = str(indicator.get("id") or "").strip()
    name = str(indicator.get("name") or indicator_id or "").strip()
    if not name:
        return None
    source_note = str(indicator.get("sourceNote") or "").strip()
    snippet = source_note[:600] if source_note else name
    url = (
        f"https://data.worldbank.org/indicator/{indicator_id}"
        if indicator_id
        else "https://data.worldbank.org/"
    )
    display = f"{indicator_id} — {name}" if indicator_id else name
    return {
        "title": display,
        "snippet": snippet,
        "full_text": source_note or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (str(indicator.get("sourceOrganization") or "World Bank").strip(),),
        "venue": "World Bank Open Data",
        "publication_date": None,
        "document_type": "statistical_indicator",
        "indicator_id": indicator_id or None,
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
        fixture = _fixture_search_path("world_bank_search_unemployment.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[WorldBank] fixture load failed: %s", exc)

    if not q:
        return {"results": []}

    terms = _query_terms(q)
    collected: list[dict[str, Any]] = []
    try:
        for page in range(1, 4):
            resp = knowledge_get(
                INDICATORS_URL,
                params={"format": "json", "per_page": 500, "page": page},
                headers=_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                break
            indicators = [
                item for item in (payload[1] or []) if isinstance(item, dict)
            ]
            ranked = sorted(
                indicators,
                key=lambda item: _indicator_score(item, terms),
                reverse=True,
            )
            for item in ranked:
                if _indicator_score(item, terms) <= 0:
                    continue
                collected.append(item)
                if len(collected) >= max_results:
                    break
            if len(collected) >= max_results:
                break
            meta = payload[0] if isinstance(payload[0], dict) else {}
            pages = int(meta.get("pages") or 1)
            if page >= pages:
                break
        return {"results": collected[: max(1, max_results)]}
    except BudgetExhaustedError:
        logger.warning("[WorldBank] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[WorldBank] indicator search failed: %s", exc)
        return {"results": []}


def search_world_bank(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search World Bank Open Data indicators by keyword."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_indicator(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
