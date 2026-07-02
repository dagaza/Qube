"""GovInfo adapter — U.S. federal publications via GovInfo Search API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_post

logger = logging.getLogger("Qube.Knowledge.GovInfo")

ADAPTER_ID = "govinfo"
RETRIEVAL_METHOD = "govinfo_search"
GOVINFO_SEARCH = "https://api.govinfo.gov/search"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_COLLECTION_LABELS = {
    "BILLS": "Congressional bill",
    "BILLSTATUS": "Bill status",
    "USCODE": "U.S. Code",
    "STATUTE": "Statutes at Large",
    "CFR": "Code of Federal Regulations",
    "FR": "Federal Register",
    "PLAW": "Public law",
    "COMPS": "Compiled Statutes",
}


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


def _api_key() -> str | None:
    return authorization_token("govinfo")


def _headers() -> dict[str, str]:
    key = _api_key()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if key:
        headers["X-Api-Key"] = key
    return headers


def _document_type(collection_code: str) -> str:
    code = (collection_code or "").strip().upper()
    if code in {"USCODE", "STATUTE", "PLAW", "COMPS"}:
        return "federal_statute"
    if code in {"CFR", "FR"}:
        return "federal_regulation"
    if code in {"BILLS", "BILLSTATUS"}:
        return "federal_bill"
    return "federal_publication"


def _row_from_result(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    collection = str(item.get("collectionCode") or "").strip().upper()
    package_id = str(item.get("packageId") or "").strip()
    date_issued = str(item.get("dateIssued") or "")[:10] or None
    author = str(item.get("governmentAuthor") or "U.S. Government").strip()
    download = item.get("download") if isinstance(item.get("download"), dict) else {}
    url = str(item.get("resultLink") or download.get("pdfLink") or "").strip()
    if not url and package_id:
        url = f"https://www.govinfo.gov/app/details/{package_id}"
    venue = _COLLECTION_LABELS.get(collection, "GovInfo")
    snippet = f"{title}. {venue}."
    if date_issued:
        snippet = f"{title} ({date_issued}). {venue}."
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": url or None,
        "_adapter": ADAPTER_ID,
        "authors": (author,),
        "venue": venue,
        "publication_date": date_issued,
        "document_type": _document_type(collection),
        "package_id": package_id or None,
        "collection_code": collection or None,
        "retrieval_method": RETRIEVAL_METHOD,
        "authority_score": 0.95,
        "jurisdiction": "US",
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("govinfo_search_privacy_act.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[GovInfo] fixture load failed: %s", exc)

    if not q or not _api_key():
        return {"results": []}

    try:
        resp = knowledge_post(
            GOVINFO_SEARCH,
            json={
                "query": q,
                "pageSize": str(max(1, min(max_results, 10))),
                "offsetMark": "*",
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": []}
    except BudgetExhaustedError:
        logger.warning("[GovInfo] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[GovInfo] search failed: %s", exc)
        return {"results": []}


def search_govinfo(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search U.S. federal publications on GovInfo."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_result(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
