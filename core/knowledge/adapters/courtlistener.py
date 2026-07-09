"""CourtListener adapter — U.S. case law search via the v4 REST API."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import knowledge_get
from core.knowledge.legal_query_planner import extract_case_name_key

logger = logging.getLogger("Qube.Knowledge.CourtListener")

ADAPTER_ID = "courtlistener"
RETRIEVAL_METHOD = "courtlistener_search"
SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
BASE_URL = "https://www.courtlistener.com"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"

_COURT_AUTHORITY = {
    "scotus": 0.95,
    "ca1": 0.88,
    "ca2": 0.88,
    "ca3": 0.88,
    "ca4": 0.88,
    "ca5": 0.88,
    "ca6": 0.88,
    "ca7": 0.88,
    "ca8": 0.88,
    "ca9": 0.88,
    "ca10": 0.88,
    "ca11": 0.88,
    "cadc": 0.88,
    "cafc": 0.88,
}


def _headers() -> dict[str, str]:
    from core.knowledge.credential_resolver import authorization_token

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = authorization_token("courtlistener")
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


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


def _authority_for_court(court_id: str | None) -> float:
    cid = (court_id or "").strip().lower()
    if cid in _COURT_AUTHORITY:
        return _COURT_AUTHORITY[cid]
    if cid.startswith("ca"):
        return 0.88
    if cid.endswith("bap") or cid.endswith("bankr"):
        return 0.72
    if "district" in cid or cid.startswith("dcd") or re.match(r"^[a-z]{2}d", cid):
        return 0.78
    return 0.82


_OPINION_TYPE_RANK: dict[str, int] = {
    "lead-opinion": 0,
    "combined-opinion": 1,
    "concurrence": 2,
    "in-part-opinion": 3,
    "dissent": 4,
}

_PROCEDURAL_SNIPPET = re.compile(
    r"\b(?:certiorari denied|in forma pauperis|motion for leave|"
    r"petition for writ of certiorari|proceedings? in forma pauperis)\b",
    re.IGNORECASE,
)


def _entry_case_name_key(entry: dict[str, Any]) -> tuple[str, str] | None:
    case_name = str(entry.get("caseName") or entry.get("caseNameFull") or "")
    return extract_case_name_key(case_name)


def _case_name_match_score(
    entry: dict[str, Any],
    query_case_key: tuple[str, str] | None,
) -> float:
    if query_case_key is None:
        return 0.0
    entry_key = _entry_case_name_key(entry)
    if entry_key is None:
        return -40.0
    if entry_key == query_case_key:
        return 120.0
    return -80.0


def _entry_rank(entry: dict[str, Any], *, query_case_key: tuple[str, str] | None) -> float:
    score = _case_name_match_score(entry, query_case_key)
    try:
        cite_count = float(entry.get("citeCount") or 0)
    except (TypeError, ValueError):
        cite_count = 0.0
    score += min(cite_count / 500.0, 80.0)
    if str(entry.get("court_id") or "").strip().lower() == "scotus":
        score += 8.0
    snippet = _opinion_snippet(entry)
    if "delivered the opinion of the Court" in snippet:
        score += 25.0
    if _PROCEDURAL_SNIPPET.search(snippet):
        score -= 35.0
    return score


def _opinion_snippet(entry: dict[str, Any]) -> str:
    opinions = [
        opinion
        for opinion in (entry.get("opinions") or [])
        if isinstance(opinion, dict)
    ]
    opinions.sort(
        key=lambda opinion: _OPINION_TYPE_RANK.get(
            str(opinion.get("type") or "").strip().lower(),
            99,
        )
    )
    for opinion in opinions:
        snippet = str(opinion.get("snippet") or "").strip()
        if not snippet:
            continue
        if _PROCEDURAL_SNIPPET.search(snippet):
            continue
        return re.sub(r"\s+", " ", snippet)[:1200]
    for opinion in opinions:
        snippet = str(opinion.get("snippet") or "").strip()
        if snippet:
            return re.sub(r"\s+", " ", snippet)[:1200]
    case_name = str(entry.get("caseNameFull") or entry.get("caseName") or "").strip()
    court = str(entry.get("court") or "").strip()
    citations = entry.get("citation") or []
    cite = citations[0] if citations else ""
    parts = [p for p in (case_name, court, str(cite).strip()) if p]
    return " — ".join(parts)[:1200]


def _absolute_url(relative: str | None) -> str | None:
    rel = str(relative or "").strip()
    if not rel:
        return None
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    return urljoin(BASE_URL, rel)


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    case_name = str(entry.get("caseName") or "").strip() or "Court opinion"
    court = str(entry.get("court") or "").strip()
    court_id = str(entry.get("court_id") or "").strip().lower()
    citations = [str(c).strip() for c in (entry.get("citation") or []) if str(c).strip()]
    snippet = _opinion_snippet(entry)
    return {
        "_adapter": ADAPTER_ID,
        "title": case_name,
        "snippet": snippet,
        "url": _absolute_url(entry.get("absolute_url")),
        "document_type": "court_opinion",
        "publication_date": entry.get("dateFiled"),
        "venue": court or "CourtListener",
        "court": court,
        "court_id": court_id,
        "citation": citations,
        "docket_number": entry.get("docketNumber"),
        "cluster_id": entry.get("cluster_id"),
        "judge": entry.get("judge"),
        "authority_score": _authority_for_court(court_id),
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
) -> dict[str, Any]:
    """Call CourtListener search API (or fixture when enabled)."""
    if _use_fixtures():
        fixture = _fixture_search_path("courtlistener_search_miranda.json")
        if fixture is not None and "miranda" in (search_query or "").lower():
            return json.loads(fixture.read_text(encoding="utf-8"))

    params = {
        "q": sanitize_api_query(search_query),
        "type": "o",
        "page_size": max(10, max_results * 4),
    }
    try:
        resp = knowledge_get(
            SEARCH_URL,
            params=params,
            headers=_headers(),
            timeout=20.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("[CourtListener] search failed: %s", exc)
    return {"results": []}


def search_courtlistener(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search CourtListener case law opinions."""
    q = sanitize_api_query(query)
    if not q:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    entries = [
        entry for entry in (payload.get("results") or []) if isinstance(entry, dict)
    ]
    query_case_key = extract_case_name_key(q)
    entries.sort(
        key=lambda entry: _entry_rank(entry, query_case_key=query_case_key),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(_row_from_entry(entry))
        if len(rows) >= max(1, max_results):
            break
    return rows
