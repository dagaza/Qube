"""FAO adapter — FAOSTAT dataset discovery (JWT bearer token required)."""

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

logger = logging.getLogger("Qube.Knowledge.FAO")

ADAPTER_ID = "fao"
RETRIEVAL_METHOD = "faostat_dataset_search"
DATASETS_URL = "https://faostatservices.fao.org/api/v1/en/data/datasets"
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
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = authorization_token("fao")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _query_terms(search_query: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in _TOKEN_SPLIT.split(sanitize_api_query(search_query).lower())
        if len(term) >= 3
    )


def _dataset_score(item: dict[str, Any], terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("code", "datasetCode", "label", "datasetName", "description")
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _extract_datasets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "datasets", "value", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _row_from_dataset(item: dict[str, Any]) -> dict[str, Any] | None:
    code = str(item.get("code") or item.get("datasetCode") or "").strip()
    label = str(item.get("label") or item.get("datasetName") or code or "").strip()
    if not label:
        return None
    description = str(item.get("description") or item.get("topic") or "").strip()
    snippet = description[:600] if description else label
    url = (
        f"https://www.fao.org/faostat/en/#data/{code}"
        if code
        else "https://www.fao.org/faostat/en/#data"
    )
    display = f"{code} — {label}" if code else label
    return {
        "title": display,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Food and Agriculture Organization of the United Nations",),
        "venue": "FAOSTAT",
        "publication_date": str(item.get("dateUpdate") or item.get("lastUpdate") or "")[:10] or None,
        "document_type": "agricultural_dataset",
        "dataset_code": code or None,
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
        fixture = _fixture_search_path("fao_search_wheat.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[FAO] fixture load failed: %s", exc)

    token = authorization_token("fao")
    if not token:
        logger.debug("[FAO] skipping live search (FAOSTAT API token required)")
        return {"datasets": []}

    terms = _query_terms(q)
    if not terms:
        return {"datasets": []}

    try:
        resp = knowledge_get(
            DATASETS_URL,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        ranked = sorted(
            _extract_datasets(payload),
            key=lambda item: _dataset_score(item, terms),
            reverse=True,
        )
        matched = [item for item in ranked if _dataset_score(item, terms) > 0]
        return {"datasets": matched[: max(1, max_results)]}
    except BudgetExhaustedError:
        logger.warning("[FAO] budget exhausted; skipping retry")
        return {"datasets": []}
    except Exception as exc:
        logger.warning("[FAO] dataset search failed: %s", exc)
        return {"datasets": []}


def search_fao(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search FAOSTAT datasets by keyword (requires FAOSTAT API bearer token)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("datasets") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_dataset(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
