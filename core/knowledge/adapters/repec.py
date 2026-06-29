"""RePEc / IDEAS adapter — economics working papers (fixture-backed stub)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.RePEc")

ADAPTER_ID = "repec"
RETRIEVAL_METHOD = "repec_search"
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
        "venue": str(entry.get("series") or entry.get("venue") or "RePEc").strip(),
        "publication_date": pub_date,
        "doi": entry.get("doi"),
        "peer_reviewed": bool(entry.get("peer_reviewed", False)),
        "preprint": bool(entry.get("preprint", True)),
        "open_access": entry.get("open_access"),
        "document_type": "working_paper",
        "repec_handle": entry.get("handle"),
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
) -> dict[str, Any]:
    """Load RePEc fixture rows or return empty (no public search API without approval)."""
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        for name in (
            "repec_search_monetary.json",
            "repec_search_economics.json",
        ):
            fixture = _fixture_search_path(name)
            if fixture is None:
                continue
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[RePEc] fixture load failed: %s", exc)
                continue
            fixture_query = str(payload.get("query") or "").lower()
            if fixture_query and fixture_query not in q.lower() and q.lower() not in fixture_query:
                continue
            return payload

    api_key = os.environ.get("QUBE_REPEC_API_KEY", "").strip()
    if not api_key:
        logger.debug(
            "[RePEc] live search unavailable (no public API; set QUBE_KNOWLEDGE_FIXTURES=1 "
            "or QUBE_REPEC_API_KEY for approved access)"
        )
        return {"results": []}

    logger.warning("[RePEc] QUBE_REPEC_API_KEY set but live client not yet implemented")
    return {"results": []}


def search_repec(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search RePEc economics literature (fixtures or future approved API)."""
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
