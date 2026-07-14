"""bioRxiv / life-science preprint adapter (fixtures + Europe PMC preprint filter)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.europe_pmc import row_from_europe_pmc_entry
from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.bioRxiv")

ADAPTER_ID = "biorxiv"
RETRIEVAL_METHOD = "biorxiv_search"
_BIORXIV_EPMC_SUFFIX = ' AND (SRC:PPR OR JOURNAL:"bioRxiv")'


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
    pub_date = str(year)[:10] if year else None
    if pub_date and len(pub_date) == 4:
        pub_date = pub_date
    excerpt = abstract[:600] if abstract else title
    doi = entry.get("doi")
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": str(entry.get("venue") or "bioRxiv").strip(),
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": bool(entry.get("peer_reviewed", False)),
        "preprint": bool(entry.get("preprint", True)),
        "open_access": entry.get("open_access", True),
        "document_type": "preprint",
        "biorxiv_id": entry.get("biorxiv_id") or entry.get("id"),
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Load fixture rows or query Europe PMC for bioRxiv preprints."""
    from core.knowledge.adapters.europe_pmc import fetch_search_results as epmc_fetch

    q = sanitize_api_query(search_query)
    if _use_fixtures():
        for name in (
            "biorxiv_search_crispr.json",
            "biorxiv_search_microbiome.json",
        ):
            fixture = _fixture_search_path(name)
            if fixture is None:
                continue
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[bioRxiv] fixture load failed: %s", exc)
                continue
            fixture_query = str(payload.get("query") or "").lower()
            if fixture_query and fixture_query not in q.lower() and q.lower() not in fixture_query:
                continue
            return payload

    if not q:
        return {"results": []}

    payload = epmc_fetch(
        q,
        max_results=max_results,
        query_suffix=_BIORXIV_EPMC_SUFFIX,
        timeout=timeout,
    )
    result_list = payload.get("resultList") or {}
    results = [
        entry
        for entry in (result_list.get("result") or [])
        if isinstance(entry, dict) and entry.get("title")
    ]
    return {"results": results, "source": "europe_pmc"}


def search_biorxiv(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search bioRxiv preprints via fixtures or Europe PMC preprint index."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    source = str(payload.get("source") or "fixture")
    entries = [
        entry for entry in (payload.get("results") or []) if isinstance(entry, dict)
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if source == "europe_pmc":
            row = row_from_europe_pmc_entry(entry, adapter_id=ADAPTER_ID)
        else:
            row = _row_from_entry(entry)
        if row and row.get("title"):
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
