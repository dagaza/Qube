"""Wikipedia MediaWiki API adapter (search + intro extract)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import requests

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.Wikipedia")

ADAPTER_ID = "wikipedia_api"
RETRIEVAL_METHOD = "api_extract"
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"


def search_wikipedia(
    query: str,
    *,
    max_results: int = 2,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Search English Wikipedia and fetch intro extracts for top hits."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    headers = {"User-Agent": USER_AGENT}
    try:
        search_resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": q,
                "srlimit": max(1, min(max_results, 5)),
                "format": "json",
                "origin": "*",
            },
            headers=headers,
            timeout=timeout,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except Exception as exc:
        logger.warning("[Wikipedia] search failed: %s", exc)
        return []

    hits = (search_data.get("query") or {}).get("search") or []
    if not hits:
        return []

    titles = [str(h.get("title") or "").strip() for h in hits if h.get("title")]
    titles = [t for t in titles if t][:max_results]
    if not titles:
        return []

    search_snippets = {
        str(h.get("title") or "").strip(): str(h.get("snippet") or "")
        for h in hits
    }

    try:
        extract_resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "redirects": 1,
                "titles": "|".join(titles),
                "format": "json",
                "origin": "*",
            },
            headers=headers,
            timeout=timeout,
        )
        extract_resp.raise_for_status()
        extract_data = extract_resp.json()
    except Exception as exc:
        logger.warning("[Wikipedia] extract failed: %s", exc)
        return []

    pages = (extract_data.get("query") or {}).get("pages") or {}
    rows: list[dict[str, Any]] = []
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "").strip()
        if not title:
            continue
        extract = str(page.get("extract") or "").strip()
        fallback_snippet = search_snippets.get(title, "")
        excerpt = extract[:600] if extract else fallback_snippet
        if not excerpt:
            continue
        page_url = str(page.get("fullurl") or "").strip()
        if not page_url:
            page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        rows.append(
            {
                "title": title,
                "snippet": excerpt,
                "full_text": extract or None,
                "url": page_url,
                "pageid": page.get("pageid"),
                "_wiki_source": True,
            }
        )
        if len(rows) >= max_results:
            break
    return rows
