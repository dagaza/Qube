"""NIST adapter — NVD keyword search (cybersecurity & standards references)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.NIST")

ADAPTER_ID = "nist"
RETRIEVAL_METHOD = "nist_nvd_keyword_search"
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
USER_AGENT = "Qube/1.0 (local@qube.app)"


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
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    api_key = authorization_token("nist")
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _row_from_cve(item: dict[str, Any]) -> dict[str, Any] | None:
    cve = item.get("cve") or {}
    cve_id = str(cve.get("id") or "").strip()
    descriptions = cve.get("descriptions") or []
    description = ""
    for desc in descriptions:
        if isinstance(desc, dict) and str(desc.get("lang") or "").lower() == "en":
            description = str(desc.get("value") or "").strip()
            break
    if not description and descriptions:
        first = descriptions[0]
        if isinstance(first, dict):
            description = str(first.get("value") or "").strip()
    title = cve_id or "NIST NVD record"
    if not description:
        description = title
    url = f"https://nvd.nist.gov/vuln/detail/{cve_id}" if cve_id else "https://nvd.nist.gov/"
    published = str(cve.get("published") or "")[:10] or None
    return {
        "title": title,
        "snippet": description[:600],
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("National Institute of Standards and Technology",),
        "venue": "NIST NVD",
        "publication_date": published,
        "document_type": "standard_reference",
        "cve_id": cve_id or None,
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
        fixture = _fixture_search_path("nist_search_encryption.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[NIST] fixture load failed: %s", exc)

    if not q:
        return {"vulnerabilities": []}

    try:
        resp = knowledge_get(
            NVD_URL,
            params={
                "keywordSearch": q,
                "resultsPerPage": max(1, min(max_results, 10)),
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"vulnerabilities": []}
    except BudgetExhaustedError:
        logger.warning("[NIST] budget exhausted; skipping retry")
        return {"vulnerabilities": []}
    except Exception as exc:
        logger.warning("[NIST] NVD search failed: %s", exc)
        return {"vulnerabilities": []}


def search_nist(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search NIST National Vulnerability Database by keyword."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("vulnerabilities") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_cve(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
