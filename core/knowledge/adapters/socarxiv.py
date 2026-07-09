"""SocArXiv preprint adapter via OSF Preprints API (no key)."""

from __future__ import annotations

import json
import logging
import os
import re
from html import unescape
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.SocArXiv")

ADAPTER_ID = "socarxiv"
RETRIEVAL_METHOD = "socarxiv_search"
OSF_PREPRINTS = "https://api.osf.io/v2/preprints/"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_TAG_RE = re.compile(r"<[^>]+>")


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


def _plain_text(value: str) -> str:
    text = unescape(_TAG_RE.sub(" ", value or ""))
    return " ".join(text.split())


def _row_from_preprint(item: dict[str, Any]) -> dict[str, Any] | None:
    attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    title = _plain_text(str(attrs.get("title") or ""))
    abstract = _plain_text(str(attrs.get("description") or ""))
    if not title and not abstract:
        return None
    preprint_id = str(item.get("id") or "").strip()
    doi = str(attrs.get("doi") or "").strip() or None
    links = attrs.get("links") if isinstance(attrs.get("links"), dict) else {}
    url = str(links.get("html") or "").strip() or None
    if not url and preprint_id:
        url = f"https://osf.io/preprints/socarxiv/{preprint_id}/"
    if not url and doi:
        url = f"https://doi.org/{doi}"
    pub_date = str(attrs.get("date_published") or attrs.get("original_publication_date") or "")[:10]
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "SocArXiv",
        "publication_date": pub_date or None,
        "doi": doi,
        "peer_reviewed": False,
        "preprint": True,
        "open_access": True,
        "document_type": "preprint",
        "preprint_id": preprint_id or None,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("socarxiv_search_inequality.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[SocArXiv] fixture load failed: %s", exc)

    if not q:
        return {"data": []}

    try:
        resp = knowledge_get(
            OSF_PREPRINTS,
            params={
                "filter[q]": q,
                "filter[provider]": "socarxiv",
                "page[size]": max(1, min(max_results, 10)),
            },
            headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.api+json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"data": []}
    except BudgetExhaustedError:
        logger.warning("[SocArXiv] budget exhausted; skipping retry")
        return {"data": []}
    except Exception as exc:
        logger.warning("[SocArXiv] search failed: %s", exc)
        return {"data": []}


def search_socarxiv(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search SocArXiv sociology preprints via OSF."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_preprint(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
