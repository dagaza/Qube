"""Crossref works search adapter (polite pool — no API key)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.Crossref")

ADAPTER_ID = "crossref"
RETRIEVAL_METHOD = "crossref_works_search"
CROSSREF_WORKS = "https://api.crossref.org/works"
USER_AGENT = "Qube/1.0 (mailto:local@qube.app)"


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


def _row_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    title_parts = item.get("title") or []
    title = str(title_parts[0] if title_parts else "").strip()
    abstract = str(item.get("abstract") or "").strip()
    if abstract.startswith("<jats:"):
        abstract = _strip_jats_abstract(abstract)
    doi_raw = item.get("DOI")
    doi = str(doi_raw).strip().lower() if doi_raw else None
    url = str(item.get("URL") or "").strip() or None
    if not url and doi:
        url = f"https://doi.org/{doi}"
    authors: list[str] = []
    for author in item.get("author") or []:
        if not isinstance(author, dict):
            continue
        given = str(author.get("given") or "").strip()
        family = str(author.get("family") or "").strip()
        name = f"{given} {family}".strip() or str(author.get("name") or "").strip()
        if name:
            authors.append(name)
    venue_parts = item.get("container-title") or item.get("short-container-title") or []
    venue = str(venue_parts[0] if venue_parts else "").strip()
    pub_date = _date_from_item(item)
    if not title and not abstract:
        return None
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": tuple(authors),
        "venue": venue or None,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": None,
        "document_type": "journal_abstract",
    }


def _strip_jats_abstract(text: str) -> str:
    import re

    cleaned = re.sub(r"<[^>]+>", " ", text)
    return " ".join(cleaned.split())


def _date_from_item(item: dict[str, Any]) -> str | None:
    for key in ("published-print", "published-online", "created", "issued"):
        block = item.get(key)
        if not isinstance(block, dict):
            continue
        parts = block.get("date-parts")
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        year = parts[0][0] if parts[0] else None
        if year:
            return str(year)
    return None


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if not q:
        return {"message": {"items": []}}

    if _use_fixtures():
        fixture = _fixture_search_path("crossref_search_climate.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[Crossref] fixture load failed: %s", exc)

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        resp = knowledge_get(
            CROSSREF_WORKS,
            params={
                "query": q,
                "rows": max(1, min(max_results, 10)),
            },
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"message": {"items": []}}
    except BudgetExhaustedError:
        logger.warning("[Crossref] budget exhausted; skipping retry")
        return {"message": {"items": []}}
    except Exception as exc:
        logger.warning("[Crossref] search failed: %s", exc)
        return {"message": {"items": []}}


def search_crossref(
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Search Crossref works metadata."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results, timeout=timeout)
    message = payload.get("message") if isinstance(payload, dict) else {}
    items = message.get("items") if isinstance(message, dict) else []
    rows: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_item(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
