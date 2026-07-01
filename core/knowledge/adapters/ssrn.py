"""SSRN working-paper adapter (fixture stub — live API not yet implemented)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.SSRN")

ADAPTER_ID = "ssrn"
RETRIEVAL_METHOD = "ssrn_search"


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


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstract") or entry.get("snippet") or "").strip()
    url = str(entry.get("url") or "").strip() or None
    authors_raw = entry.get("authors") or ()
    if isinstance(authors_raw, str):
        authors = (authors_raw.strip(),) if authors_raw.strip() else ()
    else:
        authors = tuple(str(a).strip() for a in authors_raw if str(a).strip())
    year = entry.get("year") or entry.get("publication_date")
    pub_date = str(year)[:4] if year else None
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": str(entry.get("series") or entry.get("venue") or "SSRN").strip(),
        "publication_date": pub_date,
        "doi": entry.get("doi"),
        "peer_reviewed": bool(entry.get("peer_reviewed", False)),
        "preprint": bool(entry.get("preprint", True)),
        "open_access": entry.get("open_access"),
        "document_type": "working_paper",
        "ssrn_id": entry.get("ssrn_id"),
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
) -> dict[str, Any]:
    """Load SSRN fixture rows or return empty (no public search API in v1)."""
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("ssrn_search_taylor_rule.json")
        if fixture is not None:
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
                fixture_query = str(payload.get("query") or "").lower()
                if not fixture_query or fixture_query in q.lower() or q.lower() in fixture_query:
                    return payload
            except Exception as exc:
                logger.warning("[SSRN] fixture load failed: %s", exc)

    logger.debug("[SSRN] live search unavailable (catalog stub; fixtures only)")
    return {"results": []}


def search_ssrn(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search SSRN economics working papers (fixtures only until API access)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    entries = [
        entry for entry in (payload.get("results") or []) if isinstance(entry, dict)
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = _row_from_entry(entry)
        if row.get("title"):
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
