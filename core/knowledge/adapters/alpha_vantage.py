"""Alpha Vantage adapter — market symbol search (free API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.AlphaVantage")

ADAPTER_ID = "alpha_vantage"
RETRIEVAL_METHOD = "alpha_vantage_symbol_search"
SEARCH_URL = "https://www.alphavantage.co/query"
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


def _row_from_match(match: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(match.get("1. symbol") or match.get("symbol") or "").strip()
    name = str(match.get("2. name") or match.get("name") or "").strip()
    if not symbol and not name:
        return None
    asset_type = str(match.get("3. type") or match.get("type") or "").strip()
    region = str(match.get("4. region") or match.get("region") or "").strip()
    currency = str(match.get("8. currency") or match.get("currency") or "").strip()
    timezone = str(match.get("7. timezone") or match.get("timezone") or "").strip()
    match_score = str(match.get("9. matchScore") or match.get("matchScore") or "").strip()
    snippet_parts = [
        part
        for part in (
            asset_type,
            region,
            currency,
            f"Match score {match_score}" if match_score else "",
        )
        if part
    ]
    snippet = ". ".join(snippet_parts) if snippet_parts else name or symbol
    display_title = f"{symbol} — {name}" if symbol and name else (name or symbol)
    return {
        "title": display_title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": None,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "Alpha Vantage",
        "publication_date": None,
        "document_type": "market_symbol",
        "symbol": symbol or None,
        "asset_type": asset_type or None,
        "region": region or None,
        "currency": currency or None,
        "timezone": timezone or None,
        "match_score": match_score or None,
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
        fixture = _fixture_search_path("alpha_vantage_search_microsoft.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Alpha Vantage] fixture load failed: %s", exc)

    api_key = authorization_token("alpha_vantage")
    if not api_key:
        logger.debug("[Alpha Vantage] skipping live search (API key required)")
        return {"bestMatches": []}

    if not q:
        return {"bestMatches": []}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={
                "function": "SYMBOL_SEARCH",
                "keywords": q,
                "apikey": api_key,
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            logger.warning("[Alpha Vantage] API error: %s", payload.get("Error Message"))
            return {"bestMatches": []}
        if isinstance(payload, dict) and payload.get("Note"):
            logger.warning("[Alpha Vantage] rate limit note: %s", payload.get("Note"))
            return {"bestMatches": []}
        return payload if isinstance(payload, dict) else {"bestMatches": []}
    except BudgetExhaustedError:
        logger.warning("[Alpha Vantage] budget exhausted; skipping retry")
        return {"bestMatches": []}
    except Exception as exc:
        logger.warning("[Alpha Vantage] search failed: %s", exc)
        return {"bestMatches": []}


def search_alpha_vantage(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search Alpha Vantage symbols (requires configured API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for match in payload.get("bestMatches") or []:
        if not isinstance(match, dict):
            continue
        row = _row_from_match(match)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
