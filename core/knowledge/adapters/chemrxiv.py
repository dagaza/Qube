"""ChemRxiv chemistry preprint adapter (Europe PMC ChemRxiv DOI filter)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.europe_pmc import row_from_europe_pmc_entry
from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.ChemRxiv")

ADAPTER_ID = "chemrxiv"
RETRIEVAL_METHOD = "chemrxiv_search"
_CHEMRXIV_EPMC_SUFFIX = " AND DOI:10.26434*"


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
    row = row_from_europe_pmc_entry(entry, adapter_id=ADAPTER_ID)
    if row is None:
        title = str(entry.get("title") or "").strip()
        abstract = str(entry.get("abstract") or entry.get("snippet") or "").strip()
        url = str(entry.get("url") or "").strip() or None
        return {
            "title": title,
            "snippet": (abstract or title)[:600],
            "full_text": abstract or None,
            "url": url,
            "_adapter": ADAPTER_ID,
            "authors": (),
            "venue": "ChemRxiv",
            "publication_date": entry.get("publication_date"),
            "doi": entry.get("doi"),
            "peer_reviewed": False,
            "preprint": True,
            "open_access": True,
            "document_type": "preprint",
            "chemrxiv_id": entry.get("chemrxiv_id") or entry.get("id"),
            "retrieval_method": RETRIEVAL_METHOD,
        }
    row["venue"] = "ChemRxiv"
    row["preprint"] = True
    row["peer_reviewed"] = False
    row["document_type"] = "preprint"
    row["retrieval_method"] = RETRIEVAL_METHOD
    return row


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Load fixture rows or query Europe PMC for ChemRxiv preprints."""
    from core.knowledge.adapters.europe_pmc import fetch_search_results as epmc_fetch

    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("chemrxiv_search_battery.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[ChemRxiv] fixture load failed: %s", exc)

    if not q:
        return {"results": []}

    payload = epmc_fetch(
        f"{q}{_CHEMRXIV_EPMC_SUFFIX}",
        max_results=max_results,
        timeout=timeout,
    )
    result_list = payload.get("resultList") if isinstance(payload.get("resultList"), dict) else {}
    entries = result_list.get("result") or []
    return {"results": entries if isinstance(entries, list) else []}


def search_chemrxiv(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search ChemRxiv chemistry preprints via Europe PMC DOI prefix filter."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        row = _row_from_entry(entry)
        if row.get("title") or row.get("snippet"):
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
