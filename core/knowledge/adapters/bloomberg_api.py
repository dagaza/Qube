"""Bloomberg Open API adapter — security lookup via Bloomberg HTTP bridge."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import resolve_credential
from core.knowledge.http_client import BudgetExhaustedError, knowledge_post

logger = logging.getLogger("Qube.Knowledge.Bloomberg")

ADAPTER_ID = "bloomberg_api"
RETRIEVAL_METHOD = "bloomberg_instrument_search"
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


def bloomberg_api_base_url() -> str | None:
    """Resolve Bloomberg HTTP API base URL from env or settings."""
    url = os.environ.get("QUBE_BLOOMBERG_API_URL", "").strip()
    if url:
        return url.rstrip("/")
    cred = resolve_credential("bloomberg")
    secret = (cred.secret or "").strip()
    if secret:
        return secret.rstrip("/")
    return None


def _bloomberg_host(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "bloomberg").lower()


def _row_from_instrument(item: dict[str, Any]) -> dict[str, Any] | None:
    security = str(
        item.get("security")
        or item.get("Security")
        or item.get("Ticker")
        or item.get("Parseky")
        or ""
    ).strip()
    description = str(
        item.get("description")
        or item.get("Description")
        or item.get("Name")
        or ""
    ).strip()
    if not security and not description:
        return None
    display_title = f"{security} — {description}" if security and description else (description or security)
    snippet_parts = [part for part in (security, description) if part]
    snippet = ". ".join(snippet_parts)
    return {
        "title": display_title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": None,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "Bloomberg",
        "publication_date": None,
        "document_type": "market_symbol",
        "symbol": security or None,
        "description": description or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _instruments_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    instruments: list[dict[str, Any]] = []
    for block in data:
        if not isinstance(block, dict):
            continue
        instrument_data = block.get("instrumentData")
        if isinstance(instrument_data, dict):
            rows = instrument_data.get("results") or instrument_data.get("instrument")
            if isinstance(rows, dict):
                instruments.append(rows)
            elif isinstance(rows, list):
                instruments.extend(item for item in rows if isinstance(item, dict))
        elif isinstance(instrument_data, list):
            instruments.extend(item for item in instrument_data if isinstance(item, dict))
        results = block.get("results")
        if isinstance(results, list):
            instruments.extend(item for item in results if isinstance(item, dict))
    return instruments


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("bloomberg_search_microsoft.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Bloomberg] fixture load failed: %s", exc)

    base_url = bloomberg_api_base_url()
    if not base_url or not q:
        if not base_url:
            logger.debug("[Bloomberg] skipping live search (HTTP API URL required)")
        return {"results": []}

    query = urlencode(
        {
            "ns": "blp",
            "service": "instruments",
            "type": "instrumentListRequest",
        }
    )
    request_url = f"{base_url}/request?{query}"
    body = {"query": q, "maxResults": max(1, min(max_results, 10))}
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Version": "1.0.0",
    }

    try:
        resp = knowledge_post(
            request_url,
            json=body,
            headers=headers,
            host=_bloomberg_host(base_url),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            return {"results": []}
        if payload.get("message") != "OK" and payload.get("status") not in (0, "0"):
            logger.warning("[Bloomberg] API message: %s", payload.get("message"))
        return payload
    except BudgetExhaustedError:
        logger.warning("[Bloomberg] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[Bloomberg] instrument search failed: %s", exc)
        return {"results": []}


def search_bloomberg_api(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search Bloomberg securities via a configured HTTP API bridge."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []

    if _use_fixtures() and isinstance(payload.get("results"), list):
        for entry in payload.get("results") or []:
            if not isinstance(entry, dict):
                continue
            row = _row_from_instrument(entry)
            if row is not None:
                rows.append(row)
            if len(rows) >= max_results:
                break
        return rows

    for item in _instruments_from_payload(payload):
        row = _row_from_instrument(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
