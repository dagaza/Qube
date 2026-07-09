"""Companies House adapter — UK company registry search (API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import http_basic_auth
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.CompaniesHouse")

ADAPTER_ID = "companies_house"
RETRIEVAL_METHOD = "companies_house_search"
SEARCH_URL = "https://api.company-information.service.gov.uk/search/companies"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"


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


def _row_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(item.get("title") or "").strip()
    company_number = str(item.get("company_number") or "").strip()
    if not title and not company_number:
        return None
    description = str(item.get("description") or item.get("snippet") or "").strip()
    status = str(item.get("company_status") or "").strip()
    company_type = str(item.get("company_type") or "").strip()
    created = str(item.get("date_of_creation") or "")[:10] or None
    address = item.get("address") if isinstance(item.get("address"), dict) else {}
    address_bits = [
        str(address.get(key) or "").strip()
        for key in ("address_line_1", "locality", "postal_code", "country")
    ]
    address_text = ", ".join(part for part in address_bits if part)
    snippet_parts = [part for part in (description, status, company_type, address_text) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else title
    snippet = snippet[:600]
    display_title = f"{title} ({company_number})" if company_number else title
    url = (
        f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
        if company_number
        else None
    )
    return {
        "title": display_title,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "Companies House",
        "publication_date": created,
        "document_type": "uk_company_registry",
        "company_number": company_number or None,
        "company_status": status or None,
        "company_type": company_type or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("companies_house_search_tesco.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Companies House] fixture load failed: %s", exc)

    auth = http_basic_auth("companies_house")
    if auth is None:
        logger.debug("[Companies House] skipping live search (API key required)")
        return {"items": []}

    if not q:
        return {"items": []}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={
                "q": q,
                "items_per_page": max(1, min(max_results, 10)),
            },
            headers=_headers(),
            auth=auth,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"items": []}
    except BudgetExhaustedError:
        logger.warning("[Companies House] budget exhausted; skipping retry")
        return {"items": []}
    except Exception as exc:
        logger.warning("[Companies House] search failed: %s", exc)
        return {"items": []}


def search_companies_house(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search UK Companies House registry (requires API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_item(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
