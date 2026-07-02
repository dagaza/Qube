"""NICE adapter — UK clinical guidance via syndication API (API key required)."""

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

logger = logging.getLogger("Qube.Knowledge.NICE")

ADAPTER_ID = "nice"
RETRIEVAL_METHOD = "nice_guidance_index_search"
GUIDANCE_INDEX_URL = "https://api.nice.org.uk/services/guidance/index"
USER_AGENT = "Qube/1.0 (local@qube.app)"
_ACCEPT = "application/vnd.nice.syndication.services+json"
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
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": _ACCEPT,
    }
    api_key = authorization_token("nice")
    if api_key:
        headers["API-Key"] = api_key
    return headers


def _query_terms(search_query: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in _TOKEN_SPLIT.split(sanitize_api_query(search_query).lower())
        if len(term) >= 3
    )


def _guidance_score(item: dict[str, Any], terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("title", "name", "summary", "abstract", "description", "reference", "id")
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _extract_guidance_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("guidance", "items", "entries", "results", "value", "documents"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if any(key in payload for key in ("title", "name", "summary")):
        return [payload]
    return []


def _row_from_guidance(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(
        item.get("title")
        or item.get("name")
        or item.get("documentTitle")
        or ""
    ).strip()
    reference = str(item.get("reference") or item.get("id") or item.get("guidanceId") or "").strip()
    if not title and reference:
        title = reference
    if not title:
        return None

    summary = str(
        item.get("summary")
        or item.get("abstract")
        or item.get("description")
        or item.get("shortSummary")
        or ""
    ).strip()
    snippet = summary[:600] if summary else title
    url = str(item.get("url") or item.get("webUrl") or item.get("link") or "").strip()
    if not url and reference:
        ref_slug = reference.lower().replace(" ", "")
        url = f"https://www.nice.org.uk/guidance/{ref_slug}"
    if not url:
        url = "https://www.nice.org.uk/guidance"
    display = f"{reference} — {title}" if reference and reference.lower() not in title.lower() else title
    published = str(item.get("publishedDate") or item.get("lastUpdated") or "")[:10] or None
    return {
        "title": display,
        "snippet": snippet,
        "full_text": summary or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("National Institute for Health and Care Excellence",),
        "venue": "NICE",
        "publication_date": published,
        "document_type": "clinical_guideline",
        "guidance_reference": reference or None,
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
        fixture = _fixture_search_path("nice_search_hypertension.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[NICE] fixture load failed: %s", exc)

    api_key = authorization_token("nice")
    if not api_key:
        logger.debug("[NICE] skipping live search (API key required)")
        return {"guidance": []}

    terms = _query_terms(q)
    if not terms:
        return {"guidance": []}

    try:
        resp = knowledge_get(
            GUIDANCE_INDEX_URL,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        items = _extract_guidance_items(payload)
        ranked = sorted(
            items,
            key=lambda item: _guidance_score(item, terms),
            reverse=True,
        )
        matched = [
            item
            for item in ranked
            if _guidance_score(item, terms) > 0
        ]
        return {"guidance": matched[: max(1, max_results)]}
    except BudgetExhaustedError:
        logger.warning("[NICE] budget exhausted; skipping retry")
        return {"guidance": []}
    except Exception as exc:
        logger.warning("[NICE] guidance search failed: %s", exc)
        return {"guidance": []}


def search_nice(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search NICE clinical guidance (requires syndication API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("guidance") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_guidance(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
