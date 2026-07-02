"""USDA FoodData Central adapter — food composition search."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import merge_query_params
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.USDAFDC")

ADAPTER_ID = "usda_fdc"
RETRIEVAL_METHOD = "usda_fdc_food_search"
FOODS_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
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


def _row_from_food(item: dict[str, Any]) -> dict[str, Any] | None:
    fdc_id = item.get("fdcId")
    description = str(item.get("description") or "").strip()
    if not description:
        return None
    brand = str(item.get("brandOwner") or item.get("brandName") or "").strip()
    data_type = str(item.get("dataType") or "").strip()
    snippet_parts = [part for part in (brand, data_type) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else description
    url = (
        f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{fdc_id}/nutrients"
        if fdc_id is not None
        else "https://fdc.nal.usda.gov/"
    )
    title = f"{description} ({brand})" if brand else description
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("USDA FoodData Central",),
        "venue": "USDA FDC",
        "publication_date": str(item.get("publishedDate") or "")[:10] or None,
        "document_type": "nutrition_dataset",
        "fdc_id": str(fdc_id) if fdc_id is not None else None,
        "data_type": data_type or None,
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
        fixture = _fixture_search_path("usda_fdc_search_apple.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[USDA FDC] fixture load failed: %s", exc)

    if not q:
        return {"foods": []}

    try:
        params: dict[str, Any] = {
            "query": q,
            "pageSize": max(1, min(max_results, 10)),
        }
        params = merge_query_params(params, "usda_fdc")
        if not params.get("api_key"):
            params["api_key"] = "DEMO_KEY"
        resp = knowledge_get(
            FOODS_SEARCH_URL,
            params=params,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"foods": []}
    except BudgetExhaustedError:
        logger.warning("[USDA FDC] budget exhausted; skipping retry")
        return {"foods": []}
    except Exception as exc:
        logger.warning("[USDA FDC] search failed: %s", exc)
        return {"foods": []}


def search_usda_fdc(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search USDA FoodData Central for food composition records."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("foods") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_food(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
