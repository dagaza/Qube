"""NASA Earthdata adapter — CMR collection metadata search (no key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.NASAEarthdata")

ADAPTER_ID = "nasa_earthdata"
RETRIEVAL_METHOD = "nasa_earthdata_collection_search"
SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
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


def _collection_url(entry: dict[str, Any]) -> str | None:
    for link in entry.get("links") or []:
        if not isinstance(link, dict):
            continue
        href = str(link.get("href") or "").strip()
        rel = str(link.get("rel") or "").strip().lower()
        if href and ("data#" in rel or rel.endswith("/data#") or "metadata#" in rel):
            return href
    entry_id = str(entry.get("entry_id") or entry.get("short_name") or "").strip()
    if entry_id:
        return (
            "https://search.earthdata.nasa.gov/search/granules?"
            f"p=C{entry.get('concept_id', '')}"
        )
    return "https://search.earthdata.nasa.gov/search"


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    title = str(entry.get("title") or entry.get("dataset_id") or "").strip()
    summary = str(entry.get("summary") or "").strip()
    if not title and not summary:
        return None
    snippet = summary[:600] if summary else title
    time_start = str(entry.get("time_start") or "")[:10] or None
    return {
        "title": title,
        "snippet": snippet,
        "full_text": summary or None,
        "url": _collection_url(entry),
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": str(entry.get("data_center") or "NASA Earthdata"),
        "publication_date": time_start,
        "document_type": "earthdata_collection",
        "collection_id": entry.get("entry_id") or entry.get("short_name"),
        "concept_id": entry.get("concept_id"),
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 15.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("nasa_earthdata_search_sst.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[NASA Earthdata] fixture load failed: %s", exc)

    if not q:
        return {"feed": {"entry": []}}

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={
                "keyword": q,
                "page_size": max(1, min(max_results, 10)),
            },
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"feed": {"entry": []}}
    except BudgetExhaustedError:
        logger.warning("[NASA Earthdata] budget exhausted; skipping retry")
        return {"feed": {"entry": []}}
    except Exception as exc:
        logger.warning("[NASA Earthdata] search failed: %s", exc)
        return {"feed": {"entry": []}}


def search_nasa_earthdata(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search NASA Earthdata collection metadata via CMR."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    feed = payload.get("feed") if isinstance(payload.get("feed"), dict) else {}
    entries = feed.get("entry") or []
    if isinstance(entries, dict):
        entries = [entries]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row = _row_from_entry(entry)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
