"""U.S. Census Bureau adapter — dataset discovery via data.json catalog."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.USCensus")

ADAPTER_ID = "us_census"
RETRIEVAL_METHOD = "us_census_dataset_search"
DATA_JSON_URL = "https://api.census.gov/data.json"
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


def _dataset_score(dataset: dict[str, Any], terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    haystack = " ".join(
        str(dataset.get(key) or "")
        for key in ("title", "description", "c_dataset", "programCode")
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _row_from_dataset(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    description = str(item.get("description") or "").strip()
    snippet = description[:600] if description else title
    landing = str(item.get("landingPage") or item.get("accessURL") or "").strip()
    url = landing or "https://www.census.gov/data.html"
    identifier = str(item.get("identifier") or item.get("c_dataset") or "").strip()
    return {
        "title": title,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("U.S. Census Bureau",),
        "venue": "U.S. Census Bureau",
        "publication_date": str(item.get("modified") or item.get("issued") or "")[:10] or None,
        "document_type": "statistical_release",
        "dataset_id": identifier or None,
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
        fixture = _fixture_search_path("us_census_search_population.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[US Census] fixture load failed: %s", exc)

    if not q:
        return {"results": []}

    terms = _query_terms(q)
    try:
        from core.knowledge.credential_resolver import authorization_token

        params: dict[str, Any] = {}
        census_key = authorization_token("us_census")
        if census_key:
            params["key"] = census_key
        resp = knowledge_get(
            DATA_JSON_URL,
            params=params,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        datasets = [
            item
            for item in ((payload.get("dataset") or []) if isinstance(payload, dict) else [])
            if isinstance(item, dict)
        ]
        ranked = sorted(
            datasets,
            key=lambda item: _dataset_score(item, terms),
            reverse=True,
        )
        if terms:
            ranked = [item for item in ranked if _dataset_score(item, terms) > 0]
        return {"results": ranked[: max(1, max_results)]}
    except BudgetExhaustedError:
        logger.warning("[US Census] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[US Census] dataset search failed: %s", exc)
        return {"results": []}


def search_us_census(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search U.S. Census Bureau open data catalog by keyword."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_dataset(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
