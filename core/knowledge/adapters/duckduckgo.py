"""DuckDuckGo HTML search adapter (wraps mcp.internet_tool)."""

from __future__ import annotations

from typing import Any

from mcp.internet_tool import execute_internet_search, search_internet

ADAPTER_ID = "duckduckgo"
RETRIEVAL_METHOD = "serp"


def search_duckduckgo(
    query: str,
    *,
    max_results: int = 3,
    target_site: str | None = None,
) -> list[dict[str, Any]]:
    """Return structured SERP rows from DuckDuckGo HTML search."""
    rows = search_internet(query, max_results=max_results, target_site=target_site)
    return [dict(r) for r in rows if isinstance(r, dict)]


def search_duckduckgo_detailed(
    query: str,
    *,
    max_results: int = 3,
    target_site: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return SERP rows and DDG inspection metadata for typed outcomes."""
    response = execute_internet_search(
        query,
        max_results=max_results,
        target_site=target_site,
    )
    rows = [dict(r) for r in response.rows if isinstance(r, dict)]
    inspection = dict(response.inspection) if response.inspection else None
    return rows, inspection


def is_failure_sentinel(results: list[dict[str, Any]] | None) -> bool:
    """True when DDG returned empty, no-result, or network-error sentinel rows."""
    if not results:
        return True
    snippets = " ".join(str((r or {}).get("snippet") or "") for r in results)
    if "Internet search failed" in snippets:
        return True
    if "Internet search blocked" in snippets:
        return True
    if "Internet search deferred" in snippets:
        return True
    if "No relevant internet results found" in snippets:
        return True
    return not snippets.strip()


def failure_sentinel_reason(results: list[dict[str, Any]] | None) -> str | None:
    """Return a short sentinel reason label when ``is_failure_sentinel`` is true."""
    if not is_failure_sentinel(results):
        return None
    snippets = " ".join(str((r or {}).get("snippet") or "") for r in results)
    if "Internet search blocked" in snippets:
        return "ddg_bot_challenge"
    if "Internet search deferred" in snippets:
        return "ddg_pacing_timeout"
    if "Internet search failed" in snippets:
        return "network_error"
    if "No relevant internet results found" in snippets:
        return "ddg_empty_parse"
    return "ddg_empty"
