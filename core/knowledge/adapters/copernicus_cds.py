"""Copernicus CDS adapter — climate dataset catalogue search."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.CopernicusCDS")

ADAPTER_ID = "copernicus_cds"
RETRIEVAL_METHOD = "copernicus_cds_catalogue_search"
CATALOGUE_URL = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections"
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


def _collection_score(item: dict[str, Any], terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("id", "title", "description", "keywords")
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _row_from_collection(item: dict[str, Any]) -> dict[str, Any] | None:
    collection_id = str(item.get("id") or "").strip()
    title = str(item.get("title") or collection_id or "").strip()
    if not title:
        return None
    description = str(item.get("description") or "").strip()
    snippet = description[:600] if description else title
    url = (
        f"https://cds.climate.copernicus.eu/datasets/{collection_id}"
        if collection_id
        else "https://cds.climate.copernicus.eu/"
    )
    display = f"{collection_id} — {title}" if collection_id else title
    return {
        "title": display,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Copernicus Climate Data Store",),
        "venue": "Copernicus CDS",
        "publication_date": None,
        "document_type": "climate_dataset",
        "collection_id": collection_id or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 25.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("copernicus_cds_search_temperature.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[CopernicusCDS] fixture load failed: %s", exc)

    terms = _query_terms(q)
    if not terms:
        return {"collections": []}

    try:
        resp = knowledge_get(
            CATALOGUE_URL,
            params={
                "limit": max(20, max_results * 20),
                "q": q,
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        collections = [
            item for item in (payload.get("collections") or []) if isinstance(item, dict)
        ]
        ranked = sorted(
            collections,
            key=lambda item: _collection_score(item, terms),
            reverse=True,
        )
        matched = [item for item in ranked if _collection_score(item, terms) > 0]
        if not matched:
            matched = ranked[: max(1, max_results)]
        return {"collections": matched[: max(1, max_results)]}
    except BudgetExhaustedError:
        logger.warning("[CopernicusCDS] budget exhausted; skipping retry")
        return {"collections": []}
    except Exception as exc:
        logger.warning("[CopernicusCDS] catalogue search failed: %s", exc)
        return {"collections": []}


def search_copernicus_cds(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search Copernicus Climate Data Store catalogue entries."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("collections") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_collection(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
