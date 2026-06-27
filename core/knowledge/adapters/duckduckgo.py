"""DuckDuckGo HTML search adapter (wraps mcp.internet_tool)."""

from __future__ import annotations

from typing import Any

from mcp.internet_tool import search_internet

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


def is_failure_sentinel(results: list[dict[str, Any]] | None) -> bool:
    """True when DDG returned empty, no-result, or network-error sentinel rows."""
    if not results:
        return True
    snippets = " ".join(str((r or {}).get("snippet") or "") for r in results)
    if "Internet search failed" in snippets:
        return True
    if "No relevant internet results found" in snippets:
        return True
    return not snippets.strip()
