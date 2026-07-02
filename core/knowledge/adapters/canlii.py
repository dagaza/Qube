"""CanLII adapter — Canadian case law search (API key required)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import api_key_query_params
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.CanLII")

ADAPTER_ID = "canlii"
RETRIEVAL_METHOD = "canlii_search"
SEARCH_URL = "https://api.canlii.org/v1/search/en/"
METADATA_URL = "https://api.canlii.org/v1/caseBrowse/en/{database_id}/{case_id}/"
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


def _case_id_value(raw: Any) -> str | None:
    if isinstance(raw, dict):
        for key in ("en", "fr"):
            text = str(raw.get(key) or "").strip()
            if text:
                return text
        return None
    text = str(raw or "").strip()
    return text or None


def _authority_for_database(database_id: str | None) -> float:
    db = (database_id or "").strip().lower()
    if db in {"csc-scc", "scc-csc"}:
        return 0.95
    if db.endswith("ca") or db.endswith("fc"):
        return 0.88
    return 0.84


def _row_from_case(case: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    title = str(case.get("title") or "").strip()
    citation = str(case.get("citation") or "").strip()
    database_id = str(case.get("databaseId") or "").strip()
    case_id = _case_id_value(case.get("caseId"))
    if not title and not citation:
        return None
    meta = metadata or {}
    url = str(meta.get("url") or "").strip() or None
    decision_date = str(meta.get("decisionDate") or "")[:10] or None
    keywords = str(meta.get("keywords") or "").strip()
    snippet_parts = [part for part in (citation, keywords) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else title
    return {
        "title": title,
        "snippet": snippet[:600],
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "CanLII",
        "publication_date": decision_date,
        "document_type": "court_opinion",
        "citation": [citation] if citation else [],
        "database_id": database_id or None,
        "case_id": case_id,
        "retrieval_method": RETRIEVAL_METHOD,
        "authority_score": _authority_for_database(database_id),
        "jurisdiction": "CA",
    }


def _fetch_case_metadata(
    database_id: str,
    case_id: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    if _use_fixtures():
        fixture = _fixture_search_path("canlii_case_metadata_charter.json")
        if fixture is not None:
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
            except Exception as exc:
                logger.warning("[CanLII] metadata fixture load failed: %s", exc)

    params = dict(api_key_query_params("canlii"))
    if not params:
        return {}
    url = METADATA_URL.format(database_id=database_id, case_id=case_id)
    try:
        resp = knowledge_get(url, params=params, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug("[CanLII] metadata fetch failed for %s/%s: %s", database_id, case_id, exc)
        return {}


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 15.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("canlii_search_charter.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[CanLII] fixture load failed: %s", exc)

    params = dict(api_key_query_params("canlii"))
    if not params:
        logger.debug("[CanLII] skipping live search (API key required)")
        return {"results": []}

    if not q:
        return {"results": []}

    params.update(
        {
            "fullText": q,
            "resultCount": max(1, min(max_results, 10)),
            "offset": "0",
        }
    )
    try:
        resp = knowledge_get(
            SEARCH_URL,
            params=params,
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": []}
    except BudgetExhaustedError:
        logger.warning("[CanLII] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[CanLII] search failed: %s", exc)
        return {"results": []}


def search_canlii(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search Canadian case law on CanLII (requires API key)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        case = item.get("case")
        if not isinstance(case, dict):
            continue
        database_id = str(case.get("databaseId") or "").strip()
        case_id = _case_id_value(case.get("caseId"))
        metadata: dict[str, Any] = {}
        if database_id and case_id and (api_key_query_params("canlii") or _use_fixtures()):
            metadata = _fetch_case_metadata(database_id, case_id, timeout=12.0)
        row = _row_from_case(case, metadata=metadata)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
