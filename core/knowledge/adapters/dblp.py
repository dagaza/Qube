"""DBLP publication search adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import knowledge_get

logger = logging.getLogger("Qube.Knowledge.DBLP")

ADAPTER_ID = "dblp"
RETRIEVAL_METHOD = "dblp_search"
DBLP_SEARCH = "https://dblp.org/search/publ/api"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"


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


def _parse_authors(info: dict[str, Any]) -> tuple[str, ...]:
    authors_block = info.get("authors")
    if not isinstance(authors_block, dict):
        return ()
    author = authors_block.get("author") or []
    if isinstance(author, dict):
        author = [author]
    names: list[str] = []
    for item in author:
        if isinstance(item, dict):
            name = str(item.get("text") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return tuple(names)


def _row_from_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    info = hit.get("info")
    if not isinstance(info, dict):
        return None
    title = str(info.get("title") or "").strip()
    if not title:
        return None
    venue = str(info.get("venue") or "").strip()
    year = info.get("year")
    pub_date = str(year) if year else None
    doi_raw = info.get("doi")
    doi = str(doi_raw).strip().lower() if doi_raw else None
    url = str(info.get("url") or info.get("ee") or "").strip() or None
    access = str(info.get("access") or "").strip().lower()
    doc_type = str(info.get("type") or "").strip()
    preprint = "informal" in doc_type.lower() or venue.upper() == "CORR"
    peer_reviewed = not preprint and bool(venue)
    open_access = access == "open" if access else None
    authors = _parse_authors(info)
    snippet_parts = [p for p in (venue, pub_date) if p]
    snippet = f"{title}. {' — '.join(snippet_parts)}".strip()
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": venue or None,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": peer_reviewed,
        "preprint": preprint,
        "open_access": open_access,
        "document_type": "conference_paper" if "conference" in doc_type.lower() else "journal_abstract",
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
) -> dict[str, Any]:
    """Call DBLP search API (or fixture when enabled)."""
    if _use_fixtures():
        fixture = _fixture_search_path("dblp_search_transformer.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[DBLP] fixture load failed: %s", exc)

    q = sanitize_api_query(search_query)
    if not q:
        return {"hits": {"hit": []}}

    try:
        resp = knowledge_get(
            DBLP_SEARCH,
            params={
                "q": q,
                "format": "json",
                "h": max(1, min(max_results, 10)),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=12.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("[DBLP] search failed: %s", exc)
    return {"hits": {"hit": []}}


def search_dblp(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search DBLP for computer-science publications."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    hits_block = (result or {}).get("hits") if isinstance(result, dict) else payload.get("hits")
    if not isinstance(hits_block, dict):
        hits_block = {}

    hit = hits_block.get("hit") or []
    if isinstance(hit, dict):
        hit = [hit]

    rows: list[dict[str, Any]] = []
    for item in hit:
        if not isinstance(item, dict):
            continue
        row = _row_from_hit(item)
        if row:
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
