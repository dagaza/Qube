"""USDA adapter — agricultural survey variable discovery via ERS ARMS API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_post

logger = logging.getLogger("Qube.Knowledge.USDA")

ADAPTER_ID = "usda"
RETRIEVAL_METHOD = "usda_arms_variable_search"
VARIABLE_URL = "https://api.ers.usda.gov/data/arms/variable"
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


def _api_key() -> str:
    return authorization_token("usda") or "DEMO_KEY"


def _headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Api-Key": _api_key(),
    }


def _row_from_variable(item: dict[str, Any]) -> dict[str, Any] | None:
    header = str(item.get("header") or item.get("desc") or "").strip()
    description = str(item.get("desc") or item.get("description") or "").strip()
    if not header and not description:
        return None
    title = header or description
    report = item.get("report_Dim") if isinstance(item.get("report_Dim"), dict) else {}
    report_name = str(report.get("header") or "").strip()
    snippet_parts = [part for part in (description, report_name) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else title
    variable_id = str(item.get("abb") or item.get("seq") or "").strip()
    url = "https://www.ers.usda.gov/data-products/arms-farm-financial-and-crop-production-practices/"
    display = f"{variable_id} — {title}" if variable_id and variable_id.lower() not in title.lower() else title
    return {
        "title": display,
        "snippet": snippet[:600],
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("U.S. Department of Agriculture",),
        "venue": "USDA ERS ARMS",
        "publication_date": None,
        "document_type": "agricultural_indicator",
        "variable_id": variable_id or None,
        "report_name": report_name or None,
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
        fixture = _fixture_search_path("usda_search_wheat.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[USDA] fixture load failed: %s", exc)

    if not q:
        return {"data": []}

    try:
        resp = knowledge_post(
            VARIABLE_URL,
            json={"keyword": q},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            data = payload.get("data") or []
            return {"data": data if isinstance(data, list) else []}
        return {"data": []}
    except BudgetExhaustedError:
        logger.warning("[USDA] budget exhausted; skipping retry")
        return {"data": []}
    except Exception as exc:
        logger.warning("[USDA] ARMS variable search failed: %s", exc)
        return {"data": []}


def search_usda(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search USDA ERS ARMS agricultural survey variables by keyword."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_variable(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
