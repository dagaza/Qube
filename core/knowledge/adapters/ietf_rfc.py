"""IETF RFC adapter — standards documents via IETF Datatracker API."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.IETF")

ADAPTER_ID = "ietf_rfc"
RETRIEVAL_METHOD = "ietf_datatracker_search"
DOCS_URL = "https://datatracker.ietf.org/api/v1/doc/"
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
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _rfc_url(name: str) -> str:
    match = re.search(r"rfc(\d+)", name.lower())
    if match:
        return f"https://www.rfc-editor.org/rfc/rfc{match.group(1)}.html"
    return "https://www.rfc-editor.org/"


def _row_from_doc(item: dict[str, Any]) -> dict[str, Any] | None:
    name = str(item.get("name") or "").strip()
    title = str(item.get("title") or name or "").strip()
    if not title:
        return None
    doc_type = str(item.get("doc_type") or item.get("type") or "RFC").strip()
    snippet = f"{doc_type} {name}: {title}" if name else title
    url = _rfc_url(name) if name else "https://datatracker.ietf.org/doc/"
    display = f"{name.upper()} — {title}" if name else title
    return {
        "title": display,
        "snippet": snippet[:600],
        "full_text": title,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Internet Engineering Task Force",),
        "venue": "IETF",
        "publication_date": str(item.get("time") or item.get("published") or "")[:10] or None,
        "document_type": "standard_document",
        "doc_name": name or None,
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
        fixture = _fixture_search_path("ietf_rfc_search_tls.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[IETF] fixture load failed: %s", exc)

    if not q:
        return {"objects": []}

    try:
        resp = knowledge_get(
            DOCS_URL,
            params={
                "title__icontains": q,
                "limit": max(1, min(max_results, 10)),
                "format": "json",
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            objects = payload.get("objects") or payload.get("results") or []
            return {"objects": objects if isinstance(objects, list) else []}
        return {"objects": []}
    except BudgetExhaustedError:
        logger.warning("[IETF] budget exhausted; skipping retry")
        return {"objects": []}
    except Exception as exc:
        logger.warning("[IETF] datatracker search failed: %s", exc)
        return {"objects": []}


def search_ietf_rfc(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search IETF Datatracker for RFC and Internet-Draft documents."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("objects") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_doc(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
