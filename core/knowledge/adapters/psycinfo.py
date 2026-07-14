"""PsycINFO adapter — institutional EBSCO Discovery Service (EDS) search."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import resolve_credential
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get, knowledge_post

logger = logging.getLogger("Qube.Knowledge.PsycINFO")

ADAPTER_ID = "psycinfo"
RETRIEVAL_METHOD = "psycinfo_eds_search"
EBSCO_AUTH_URL = "https://eds-api.ebscohost.com/authservice/rest/uidauth"
EBSCO_SESSION_URL = "https://eds-api.ebscohost.com/edsapi/rest/createsession"
EBSCO_SEARCH_URL = "https://eds-api.ebscohost.com/edsapi/rest/search"
USER_AGENT = "Qube/1.0 (local@qube.app)"

_AUTH_CACHE: dict[str, Any] = {"auth_token": None, "session_token": None, "expires_at": 0.0}


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


def _ebsco_auth_parts() -> tuple[str | None, str | None, str]:
    """Resolve EDS user id, password, and profile name."""
    user_id = os.environ.get("QUBE_EBSCO_EDS_USER_ID", "").strip()
    password = os.environ.get("QUBE_EBSCO_EDS_PASSWORD", "").strip()
    profile = os.environ.get("QUBE_EBSCO_EDS_PROFILE", "eds").strip() or "eds"

    cred = resolve_credential("ebsco_eds")
    secret = (cred.secret or "").strip()
    if secret:
        if "|" in secret:
            parts = [part.strip() for part in secret.split("|")]
            user_id = user_id or (parts[0] if parts else "")
            password = password or (parts[1] if len(parts) > 1 else "")
            if len(parts) > 2 and parts[2]:
                profile = parts[2]
        elif not password:
            password = secret

    return user_id or None, password or None, profile


def _eds_headers(auth_token: str, session_token: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-authenticationToken": auth_token,
    }
    if session_token:
        headers["x-sessionToken"] = session_token
    return headers


def _authenticate_eds(*, timeout: float) -> tuple[str | None, str | None]:
    now = time.time()
    cached_auth = _AUTH_CACHE.get("auth_token")
    cached_session = _AUTH_CACHE.get("session_token")
    expires_at = float(_AUTH_CACHE.get("expires_at") or 0.0)
    if cached_auth and cached_session and now < expires_at:
        return str(cached_auth), str(cached_session)

    user_id, password, profile = _ebsco_auth_parts()
    if not user_id or not password:
        return None, None

    try:
        auth_resp = knowledge_post(
            EBSCO_AUTH_URL,
            json={"UserId": user_id, "Password": password, "InterfaceId": "WSapi"},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
        )
        auth_resp.raise_for_status()
        auth_payload = auth_resp.json()
        auth_token = str(auth_payload.get("AuthToken") or "").strip()
        auth_timeout = int(auth_payload.get("AuthTimeout") or 1800)
        if not auth_token:
            return None, None

        session_resp = knowledge_post(
            EBSCO_SESSION_URL,
            json={"Profile": profile, "Guest": "n"},
            headers=_eds_headers(auth_token),
            timeout=timeout,
        )
        session_resp.raise_for_status()
        session_payload = session_resp.json()
        session_token = str(session_payload.get("SessionToken") or "").strip()
        if not session_token:
            return None, None

        ttl = max(60, min(auth_timeout - 60, 1700))
        _AUTH_CACHE.update(
            {
                "auth_token": auth_token,
                "session_token": session_token,
                "expires_at": now + ttl,
            }
        )
        return auth_token, session_token
    except Exception as exc:
        logger.warning("[PsycINFO] EDS authentication failed: %s", exc)
        return None, None


def _extract_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        for key in ("Name", "Title", "Text", "value", "Value"):
            if key in node:
                text = _extract_text(node.get(key))
                if text:
                    return text
        return ""
    if isinstance(node, list):
        parts = [_extract_text(item) for item in node]
        return " ".join(part for part in parts if part).strip()
    return str(node).strip()


def _row_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    record_info = record.get("RecordInfo") if isinstance(record.get("RecordInfo"), dict) else {}
    bib = record_info.get("BibRecord") if isinstance(record_info.get("BibRecord"), dict) else {}
    entity = bib.get("BibEntity") if isinstance(bib.get("BibEntity"), dict) else {}

    titles = entity.get("Titles")
    title = _extract_text(titles)
    if not title and isinstance(titles, list) and titles:
        title = _extract_text(titles[0])

    abstract = _extract_text(entity.get("Abstracts"))
    if not title and not abstract:
        return None

    authors_raw = entity.get("Authors")
    authors: list[str] = []
    if isinstance(authors_raw, list):
        for author in authors_raw:
            name = _extract_text(author)
            if name:
                authors.append(name)
    elif isinstance(authors_raw, dict):
        author_items = authors_raw.get("Author")
        if isinstance(author_items, dict):
            author_items = [author_items]
        if isinstance(author_items, list):
            for author in author_items:
                name = _extract_text(author)
                if name:
                    authors.append(name)

    pub_date = _extract_text(entity.get("Dates"))
    if pub_date and len(pub_date) >= 4:
        pub_date = pub_date[:4]

    identifiers = entity.get("Identifiers")
    doi = None
    url = None
    if isinstance(identifiers, list):
        for ident in identifiers:
            if not isinstance(ident, dict):
                continue
            source = str(ident.get("Source") or "").strip().lower()
            value = str(ident.get("Value") or "").strip()
            if source == "doi" and value:
                doi = value.lower()
            elif source in {"url", "plink"} and value:
                url = value

    header = record.get("Header") if isinstance(record.get("Header"), dict) else {}
    db_label = str(header.get("DbLabel") or "PsycINFO").strip()
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": tuple(authors),
        "venue": db_label,
        "publication_date": pub_date or None,
        "doi": doi,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": None,
        "document_type": "journal_abstract",
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _row_from_fixture(entry: dict[str, Any]) -> dict[str, Any] | None:
    title = str(entry.get("title") or "").strip()
    if not title:
        return None
    abstract = str(entry.get("abstract") or entry.get("snippet") or "").strip()
    authors_raw = entry.get("authors") or ()
    if isinstance(authors_raw, str):
        authors = (authors_raw.strip(),) if authors_raw.strip() else ()
    else:
        authors = tuple(str(a).strip() for a in authors_raw if str(a).strip())
    year = entry.get("year") or entry.get("publication_date")
    pub_date = str(year)[:4] if year else None
    return {
        "title": title,
        "snippet": (abstract or title)[:600],
        "full_text": abstract or None,
        "url": str(entry.get("url") or "").strip() or None,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": str(entry.get("venue") or "PsycINFO").strip(),
        "publication_date": pub_date,
        "doi": entry.get("doi"),
        "peer_reviewed": bool(entry.get("peer_reviewed", True)),
        "preprint": bool(entry.get("preprint", False)),
        "open_access": entry.get("open_access"),
        "document_type": "journal_abstract",
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("psycinfo_search_cognitive_load.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[PsycINFO] fixture load failed: %s", exc)

    if not q:
        return {"results": []}

    auth_token, session_token = _authenticate_eds(timeout=timeout)
    if not auth_token or not session_token:
        logger.debug("[PsycINFO] skipping live search (EBSCO EDS credentials required)")
        return {"results": []}

    params = {
        "query-1": q,
        "view": "detailed",
        "recordsperpage": str(max(1, min(max_results, 10))),
        "highlight": "n",
    }
    database = os.environ.get("QUBE_EBSCO_EDS_DATABASE", "").strip()
    if database:
        params["action"] = f"addfacetfilter(Database:{database})"

    try:
        resp = knowledge_get(
            f"{EBSCO_SEARCH_URL}?{urlencode(params)}",
            headers=_eds_headers(auth_token, session_token),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": []}
    except BudgetExhaustedError:
        logger.warning("[PsycINFO] EDS budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[PsycINFO] EDS search failed: %s", exc)
        return {"results": []}


def _records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]

    search_result = payload.get("SearchResult")
    if not isinstance(search_result, dict):
        return []
    data = search_result.get("Data")
    if not isinstance(data, dict):
        return []
    records = data.get("Records")
    if isinstance(records, dict):
        record = records.get("Record")
        if isinstance(record, dict):
            return [record]
        if isinstance(record, list):
            return [item for item in record if isinstance(item, dict)]
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    return []


def search_psycinfo(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search PsycINFO via institutional EBSCO EDS credentials."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []

    if _use_fixtures() and isinstance(payload.get("results"), list):
        first = payload["results"][0] if payload["results"] else {}
        if isinstance(first, dict) and "title" in first and "RecordInfo" not in first:
            for entry in payload.get("results") or []:
                if not isinstance(entry, dict):
                    continue
                row = _row_from_fixture(entry)
                if row is not None:
                    rows.append(row)
                if len(rows) >= max_results:
                    break
            return rows

    for record in _records_from_payload(payload):
        row = _row_from_record(record)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
