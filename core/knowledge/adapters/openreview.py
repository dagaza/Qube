"""OpenReview conference submission search adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.OpenReview")

ADAPTER_ID = "openreview"
RETRIEVAL_METHOD = "openreview_note_search"
OPENREVIEW_SEARCH = "https://api2.openreview.net/notes/search"
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


def _content_value(content: dict[str, Any], key: str) -> Any:
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw.get("value")
    return raw


def _normalize_authors(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        name = raw.strip()
        return (name,) if name else ()
    if not isinstance(raw, list):
        return ()
    names: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name and not name.startswith("http"):
            names.append(name)
    return tuple(names)


def _publication_year(note: dict[str, Any], content: dict[str, Any]) -> str | None:
    pdate = note.get("pdate") or note.get("cdate")
    if isinstance(pdate, (int, float)) and pdate > 0:
        return str(int(pdate))[:4]
    venue = str(_content_value(content, "venue") or "").strip()
    for token in venue.split():
        if token.isdigit() and len(token) == 4:
            return token
    return None


def _pick_url(note: dict[str, Any], content: dict[str, Any]) -> str | None:
    for key in ("pdf", "html"):
        value = _content_value(content, key)
        if isinstance(value, str) and value.strip().startswith("http"):
            return value.strip()
    forum = str(note.get("forum") or note.get("id") or "").strip()
    if forum:
        return f"https://openreview.net/forum?id={forum}"
    return None


def _row_from_note(note: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(note, dict):
        return None
    content = note.get("content")
    if not isinstance(content, dict):
        return None
    title = str(_content_value(content, "title") or "").strip()
    abstract = str(_content_value(content, "abstract") or "").strip()
    if not title and not abstract:
        return None
    venue = str(_content_value(content, "venue") or "").strip() or None
    authors = _normalize_authors(_content_value(content, "authors"))
    pub_date = _publication_year(note, content)
    url = _pick_url(note, content)
    pdf_url = str(_content_value(content, "pdf") or "").strip().lower()
    venue_lower = (venue or "").lower()
    preprint = (
        "corr" in venue_lower
        or "arxiv" in pdf_url
        or "dblp.org/journals/corr" in str(_content_value(content, "venueid") or "").lower()
    )
    peer_reviewed = not preprint
    doc_type = "conference_paper" if peer_reviewed else "preprint"
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": venue or "OpenReview",
        "publication_date": pub_date,
        "peer_reviewed": peer_reviewed,
        "preprint": preprint,
        "open_access": True,
        "document_type": doc_type,
        "openreview_forum_id": str(note.get("forum") or note.get("id") or "").strip() or None,
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
        fixture = _fixture_search_path("openreview_search_transformer.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[OpenReview] fixture load failed: %s", exc)

    if not q:
        return {"notes": []}

    try:
        resp = knowledge_get(
            OPENREVIEW_SEARCH,
            params={
                "term": q,
                "source": "forum",
                "limit": max(1, min(max_results, 25)),
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"notes": []}
    except BudgetExhaustedError:
        logger.warning("[OpenReview] budget exhausted; skipping retry")
        return {"notes": []}
    except Exception as exc:
        logger.warning("[OpenReview] note search failed: %s", exc)
        return {"notes": []}


def search_openreview(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search OpenReview submissions (ICLR, NeurIPS, ICML, and related venues)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    notes = payload.get("notes") or []
    rows: list[dict[str, Any]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        row = _row_from_note(note)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
