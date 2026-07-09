"""ACL Anthology search adapter via Verbatim metadata API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.ACLAnthology")

ADAPTER_ID = "acl_anthology"
RETRIEVAL_METHOD = "acl_anthology_verbatim_search"
VERBATIM_SEARCH = "https://verbatim.krlabs.eu/api/v1/papers/search"
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


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _normalize_authors(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        name = raw.strip()
        return (name,) if name else ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(item or "").strip() for item in raw if str(item or "").strip())


def _venue_label(item: dict[str, Any]) -> str:
    booktitle = str(item.get("booktitle") or "").strip()
    if booktitle:
        return booktitle
    venue = item.get("venue")
    if isinstance(venue, list):
        joined = " / ".join(str(v).strip() for v in venue if str(v).strip())
        if joined:
            return joined
    if isinstance(venue, str) and venue.strip():
        return venue.strip()
    return "ACL Anthology"


def _row_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    abstract = str(item.get("abstract") or "").strip()
    snippet_raw = str(item.get("snippet") or "").strip()
    body = abstract or snippet_raw
    url = str(item.get("url") or "").strip()
    if not url:
        paper_id = str(item.get("id") or "").strip()
        if paper_id:
            url = f"https://aclanthology.org/{paper_id}/"
    year = item.get("year")
    pub_date = str(year) if year else None
    authors = _normalize_authors(item.get("authors"))
    venue = _venue_label(item)
    excerpt = body[:600] if body else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": body or None,
        "url": url or None,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": venue,
        "publication_date": pub_date,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": True,
        "document_type": "conference_paper",
        "anthology_id": str(item.get("id") or "").strip() or None,
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
        fixture = _fixture_search_path("acl_anthology_search_transformer.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[ACLAnthology] fixture load failed: %s", exc)

    if not q:
        return {"items": []}

    try:
        resp = knowledge_get(
            VERBATIM_SEARCH,
            params={
                "query": q,
                "collection_ids": "anthology",
                "limit": max(1, min(max_results, 25)),
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"items": []}
    except BudgetExhaustedError:
        logger.warning("[ACLAnthology] budget exhausted; skipping retry")
        return {"items": []}
    except Exception as exc:
        logger.warning("[ACLAnthology] search failed: %s", exc)
        return {"items": []}


def search_acl_anthology(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search ACL Anthology proceedings via Verbatim metadata search."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    items = payload.get("items") or []
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = _row_from_item(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
