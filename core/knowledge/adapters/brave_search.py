"""Brave Search API adapter (JSON web search)."""

from __future__ import annotations

import logging
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credentials import resolve_credential
from core.knowledge.http_client import knowledge_get

logger = logging.getLogger("Qube.Knowledge.BraveSearch")

ADAPTER_ID = "brave_search"
RETRIEVAL_METHOD = "api_search"
BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"


def brave_search_configured() -> bool:
    """True when a Brave Search API key is available (settings or env)."""
    return bool((resolve_credential(ADAPTER_ID).secret or "").strip())


def search_brave(
    query: str,
    *,
    max_results: int = 5,
    target_site: str | None = None,
    timeout: float = 8.0,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Search Brave Web API; returns (rows, inspection metadata)."""
    scoped_query = query
    if target_site:
        scoped_query = f"site:{target_site} {query}"
    q = sanitize_api_query(scoped_query)
    if not q or max_results <= 0:
        return [], {"response_kind": "no_results", "http_status": None, "parsed_rows": 0}

    secret = (resolve_credential(ADAPTER_ID).secret or "").strip()
    if not secret:
        return [], {
            "response_kind": "no_credentials",
            "http_status": None,
            "parsed_rows": 0,
        }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": secret,
    }
    try:
        response = knowledge_get(
            BRAVE_WEB_SEARCH_URL,
            params={
                "q": q,
                "count": max(1, min(max_results, 20)),
                "search_lang": "en",
            },
            headers=headers,
            timeout=timeout,
        )
        http_status = response.status_code
        if http_status == 401 or http_status == 403:
            logger.warning("[BraveSearch] unauthorized (http=%s)", http_status)
            return [], {
                "response_kind": "auth_error",
                "http_status": http_status,
                "parsed_rows": 0,
            }
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("[BraveSearch] search failed: %s", exc)
        return [], {
            "response_kind": "network_error",
            "http_status": None,
            "parsed_rows": 0,
            "error": str(exc),
        }

    web_block = payload.get("web") if isinstance(payload, dict) else None
    raw_results = (web_block or {}).get("results") if isinstance(web_block, dict) else None
    if not isinstance(raw_results, list):
        raw_results = []

    rows: list[dict[str, Any]] = []
    for item in raw_results[:max_results]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        rows.append(
            {
                "title": title,
                "snippet": description,
                "url": url,
            }
        )

    inspection = {
        "response_kind": "serp" if rows else "empty_parse",
        "http_status": http_status,
        "parsed_rows": len(rows),
    }
    logger.info(
        "[BraveSearch] query=%r http_status=%s parsed_rows=%d",
        q[:120],
        http_status,
        len(rows),
    )
    return rows, inspection
