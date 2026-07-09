"""NOAA adapter — climate dataset metadata via NCEI CDO API (token required)."""

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

logger = logging.getLogger("Qube.Knowledge.NOAA")

ADAPTER_ID = "noaa"
RETRIEVAL_METHOD = "noaa_dataset_search"
DATASETS_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
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
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = authorization_token("noaa")
    if token:
        headers["token"] = token
    return headers


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
        for key in ("id", "name", "uid", "description")
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _row_from_dataset(dataset: dict[str, Any]) -> dict[str, Any] | None:
    dataset_id = str(dataset.get("id") or "").strip()
    name = str(dataset.get("name") or dataset_id or "").strip()
    if not name:
        return None
    mindate = str(dataset.get("mindate") or "")[:10] or None
    maxdate = str(dataset.get("maxdate") or "")[:10] or None
    coverage = dataset.get("datacoverage")
    snippet_parts = [
        part
        for part in (
            f"Dataset ID {dataset_id}" if dataset_id else "",
            f"Coverage from {mindate}" if mindate else "",
            f"through {maxdate}" if maxdate else "",
            f"Data coverage score {coverage}" if coverage is not None else "",
        )
        if part
    ]
    snippet = ". ".join(snippet_parts) if snippet_parts else name
    url = (
        f"https://www.ncei.noaa.gov/access/search/data-search/dataset/{dataset_id}"
        if dataset_id
        else "https://www.ncei.noaa.gov/access/search/data-search"
    )
    return {
        "title": name,
        "snippet": snippet[:600],
        "full_text": None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "NOAA NCEI",
        "publication_date": maxdate,
        "document_type": "environmental_dataset",
        "dataset_id": dataset_id or None,
        "mindate": mindate,
        "maxdate": maxdate,
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
        fixture = _fixture_search_path("noaa_search_temperature.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[NOAA] fixture load failed: %s", exc)

    if not authorization_token("noaa"):
        logger.debug("[NOAA] skipping live search (API token required)")
        return {"results": []}

    if not q:
        return {"results": []}

    try:
        resp = knowledge_get(
            DATASETS_URL,
            params={"limit": 100},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        datasets = [
            item for item in (payload.get("results") or []) if isinstance(item, dict)
        ]
        terms = _query_terms(q)
        ranked = sorted(
            datasets,
            key=lambda item: _dataset_score(item, terms),
            reverse=True,
        )
        if terms:
            ranked = [item for item in ranked if _dataset_score(item, terms) > 0]
        return {"results": ranked[: max(1, max_results)]}
    except BudgetExhaustedError:
        logger.warning("[NOAA] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[NOAA] dataset search failed: %s", exc)
        return {"results": []}


def search_noaa(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search NOAA NCEI climate datasets (requires API token)."""
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
