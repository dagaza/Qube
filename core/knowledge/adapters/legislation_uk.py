"""UK legislation adapter — statutes via legislation.gov.uk Atom search feed."""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.LegislationUK")

ADAPTER_ID = "legislation_uk"
RETRIEVAL_METHOD = "legislation_uk_search"
LEGISLATION_SEARCH = "https://www.legislation.gov.uk/search/data.feed"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_ATOM_NS = "http://www.w3.org/2005/Atom"


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
    return {"User-Agent": USER_AGENT, "Accept": "application/atom+xml"}


def _atom_text(parent: ET.Element, tag: str) -> str:
    element = parent.find(f"{{{_ATOM_NS}}}{tag}")
    return str(element.text or "").strip() if element is not None else ""


def _public_url(href: str) -> str:
    url = href.strip()
    if not url:
        return ""
    url = url.replace("http://", "https://").replace("/id/", "/")
    if url.endswith("/id"):
        url = url[:-3]
    return url


def _atom_link(entry: ET.Element) -> str | None:
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        rel = str(link.get("rel") or "alternate").strip().lower()
        href = str(link.get("href") or "").strip()
        if href and rel in {"alternate", "self"}:
            public = _public_url(href)
            if public and "/id/" not in public:
                return public
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        href = str(link.get("href") or "").strip()
        if href:
            return _public_url(href) or None
    return None


def _row_from_entry(entry: ET.Element) -> dict[str, Any] | None:
    title = _atom_text(entry, "title")
    if not title:
        return None
    summary = _atom_text(entry, "summary")
    published = _atom_text(entry, "published") or _atom_text(entry, "updated")
    pub_date = published[:10] if published else None
    url = _atom_link(entry)
    snippet = summary[:600] if summary else title[:600]
    return {
        "title": title,
        "snippet": snippet,
        "full_text": summary or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("UK Legislation",),
        "venue": "legislation.gov.uk",
        "publication_date": pub_date,
        "document_type": "uk_legislation",
        "retrieval_method": RETRIEVAL_METHOD,
        "authority_score": 0.96,
        "jurisdiction": "UK",
    }


def _rows_from_feed(body: str, *, max_results: int) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        logger.warning("[LegislationUK] Atom parse failed: %s", exc)
        return []
    rows: list[dict[str, Any]] = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        row = _row_from_entry(entry)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("legislation_uk_search_data_protection.xml")
        if fixture is not None:
            try:
                return {"feed": fixture.read_text(encoding="utf-8")}
            except Exception as exc:
                logger.warning("[LegislationUK] fixture load failed: %s", exc)

    if not q:
        return {"feed": ""}

    try:
        resp = knowledge_get(
            LEGISLATION_SEARCH,
            params={
                "title": q,
                "results-count": max(1, min(max_results, 10)),
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return {"feed": resp.text}
    except BudgetExhaustedError:
        logger.warning("[LegislationUK] budget exhausted; skipping retry")
        return {"feed": ""}
    except Exception as exc:
        logger.warning("[LegislationUK] search failed: %s", exc)
        return {"feed": ""}


def search_legislation_uk(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search UK legislation by title via legislation.gov.uk Atom feed."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    body = str(payload.get("feed") or "")
    if not body:
        return []
    return _rows_from_feed(body, max_results=max_results)
