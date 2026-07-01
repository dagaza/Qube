"""RePEc / IDEAS adapter — economics literature via EconBiz (live) or fixtures."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.RePEc")

ADAPTER_ID = "repec"
RETRIEVAL_METHOD = "repec_search"
ECONBIZ_SEARCH = "https://api.econbiz.de/v1/search"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_REPEC_HANDLE_FROM_URL = re.compile(
    r"ideas\.repec\.org/(?:[a-z]/)+([^/?#]+(?:/[^/?#]+)*)",
    re.IGNORECASE,
)


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


def _pick_url(identifier_urls: list[Any]) -> str | None:
    for raw in identifier_urls:
        url = str(raw or "").strip()
        if not url:
            continue
        if "ideas.repec.org" in url.lower():
            return url
    for raw in identifier_urls:
        url = str(raw or "").strip()
        if url:
            return url
    return None


def _repec_handle_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = _REPEC_HANDLE_FROM_URL.search(url)
    if not match:
        return None
    path = match.group(1).replace("/", ":")
    if path.lower().endswith(".html"):
        path = path[:-5]
    return f"RePEc:{path}" if path else None


def _first_str(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return None
    text = str(value or "").strip()
    return text or None


def _authors_from_hit(hit: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for key in ("creator", "person", "contributor"):
        block = hit.get(key) or []
        if isinstance(block, str):
            block = [block]
        for item in block:
            name = str(item or "").strip()
            if name and name not in names:
                names.append(name)
    return tuple(names)


def _is_working_paper(hit: dict[str, Any]) -> bool:
    genres = hit.get("type_genre") or []
    if isinstance(genres, str):
        genres = [genres]
    joined = " ".join(str(g).lower() for g in genres)
    return any(
        token in joined
        for token in (
            "working paper",
            "arbeitspapier",
            "discussion paper",
            "grey literature",
            "graue literatur",
        )
    )


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
    pub_date = str(year)[:4] if year else None
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": authors,
        "venue": str(entry.get("series") or entry.get("venue") or "RePEc").strip(),
        "publication_date": pub_date,
        "doi": entry.get("doi"),
        "peer_reviewed": bool(entry.get("peer_reviewed", False)),
        "preprint": bool(entry.get("preprint", True)),
        "open_access": entry.get("open_access"),
        "document_type": "working_paper",
        "repec_handle": entry.get("handle"),
    }


def _row_from_econbiz_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    title = str(hit.get("title") or "").strip()
    if not title:
        return None
    identifier_urls = hit.get("identifier_url") or []
    if isinstance(identifier_urls, str):
        identifier_urls = [identifier_urls]
    url = _pick_url(list(identifier_urls))
    series = _first_str(hit.get("series"))
    is_part_of = _first_str(hit.get("isPartOf"))
    venue = series or is_part_of or _first_str(hit.get("publisher")) or "RePEc / EconBiz"
    pub_date = _first_str(hit.get("date"))
    if pub_date:
        pub_date = pub_date.strip("[]")[:4]
    subjects = hit.get("subject") or []
    if isinstance(subjects, str):
        subjects = [subjects]
    subject_text = ", ".join(str(s).strip() for s in subjects[:6] if str(s).strip())
    snippet_parts = [p for p in (venue, pub_date, subject_text) if p]
    snippet = f"{title}. {' — '.join(snippet_parts)}".strip()
    preprint = _is_working_paper(hit)
    peer_reviewed = not preprint and bool(is_part_of)
    doi = None
    for raw in identifier_urls:
        link = str(raw or "").strip()
        if link.lower().startswith("https://doi.org/"):
            doi = link.split("doi.org/", 1)[-1].strip().lower()
            break
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": _authors_from_hit(hit),
        "venue": venue,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": peer_reviewed,
        "preprint": preprint,
        "open_access": None,
        "document_type": "working_paper" if preprint else "journal_abstract",
        "repec_handle": _repec_handle_from_url(url),
        "econbiz_id": hit.get("id"),
    }


def _fetch_econbiz_live(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if not q:
        return {"hits": {"hits": []}}
    try:
        resp = requests.get(
            ECONBIZ_SEARCH,
            params={"q": q, "size": max(1, min(max_results, 10))},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("[RePEc] EconBiz search failed: %s", exc)
    return {"hits": {"hits": []}}


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
) -> dict[str, Any]:
    """Load fixtures, query EconBiz (live), or use approved RePEc API when configured."""
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        for name in (
            "repec_search_monetary.json",
            "repec_search_economics.json",
        ):
            fixture = _fixture_search_path(name)
            if fixture is None:
                continue
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[RePEc] fixture load failed: %s", exc)
                continue
            fixture_query = str(payload.get("query") or "").lower()
            if fixture_query and fixture_query not in q.lower() and q.lower() not in fixture_query:
                continue
            return payload

    api_key = os.environ.get("QUBE_REPEC_API_KEY", "").strip()
    if api_key:
        logger.debug(
            "[RePEc] QUBE_REPEC_API_KEY set but official search API unavailable; "
            "using EconBiz discovery"
        )

    return _fetch_econbiz_live(q, max_results=max_results)


def search_repec(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search economics literature indexed via RePEc/IDEAS (EconBiz live discovery)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    if isinstance(payload.get("results"), list):
        entries = [
            entry for entry in (payload.get("results") or []) if isinstance(entry, dict)
        ]
        rows: list[dict[str, Any]] = []
        for entry in entries:
            row = _row_from_entry(entry)
            if row.get("title"):
                rows.append(row)
            if len(rows) >= max(1, max_results):
                break
        return rows

    hits_block = payload.get("hits") if isinstance(payload.get("hits"), dict) else {}
    hits = hits_block.get("hits") or []
    if isinstance(hits, dict):
        hits = [hits]

    rows: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        row = _row_from_econbiz_hit(hit)
        if row:
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
