"""INSPIRE-HEP adapter — open high-energy / astrophysics literature (live REST API)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import knowledge_get

logger = logging.getLogger("Qube.Knowledge.INSPIRE")

ADAPTER_ID = "inspire_hep"
RETRIEVAL_METHOD = "inspire_literature_search"
INSPIRE_LITERATURE = "https://inspirehep.net/api/literature"
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


def _first_title(metadata: dict[str, Any]) -> str:
    titles = metadata.get("titles") or []
    if isinstance(titles, dict):
        titles = [titles]
    for item in titles:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
        else:
            title = str(item or "").strip()
        if title:
            return title
    return ""


def _abstract_text(metadata: dict[str, Any]) -> str:
    abstracts = metadata.get("abstracts")
    if isinstance(abstracts, dict):
        return str(abstracts.get("value") or "").strip()
    if isinstance(abstracts, list):
        for item in abstracts:
            if isinstance(item, dict):
                text = str(item.get("value") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                return text
    return ""


def _authors(metadata: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for author in metadata.get("authors") or []:
        if not isinstance(author, dict):
            continue
        name = str(author.get("full_name") or "").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _venue(metadata: dict[str, Any]) -> str | None:
    pub_info = metadata.get("publication_info") or []
    if isinstance(pub_info, dict):
        pub_info = [pub_info]
    for item in pub_info:
        if not isinstance(item, dict):
            continue
        journal = str(item.get("journal_title") or item.get("journal") or "").strip()
        if journal:
            return journal
    return None


def _publication_date(metadata: dict[str, Any]) -> str | None:
    pub_info = metadata.get("publication_info") or []
    if isinstance(pub_info, dict):
        pub_info = [pub_info]
    for item in pub_info:
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        if year:
            return str(year)[:4]
    return None


def _pick_url(metadata: dict[str, Any], *, record_id: str | None) -> str | None:
    for block in metadata.get("arxiv_eprints") or []:
        if not isinstance(block, dict):
            continue
        eprint = str(block.get("value") or "").strip()
        if eprint:
            return f"https://arxiv.org/abs/{eprint}"
    doi = metadata.get("doi")
    if isinstance(doi, str) and doi.strip():
        value = doi.strip()
        return value if value.startswith("http") else f"https://doi.org/{value}"
    if record_id:
        return f"https://inspirehep.net/literature/{record_id}"
    control = metadata.get("control_number")
    if control:
        return f"https://inspirehep.net/literature/{control}"
    return None


def _is_preprint(metadata: dict[str, Any]) -> bool:
    if metadata.get("arxiv_eprints"):
        return True
    doc_type = str(metadata.get("document_type") or "").lower()
    if doc_type in {"thesis", "preprint", "report"}:
        return True
    for keyword in metadata.get("keywords") or []:
        if not isinstance(keyword, dict):
            continue
        value = str(keyword.get("value") or "").lower()
        if value in {"thesis", "preprint"}:
            return True
    return False


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
        "venue": str(entry.get("venue") or "INSPIRE-HEP").strip(),
        "publication_date": pub_date,
        "doi": entry.get("doi"),
        "peer_reviewed": bool(entry.get("peer_reviewed", False)),
        "preprint": bool(entry.get("preprint", True)),
        "open_access": entry.get("open_access", True),
        "document_type": entry.get("document_type") or "literature",
        "inspire_recid": entry.get("inspire_recid"),
        "citation_count": entry.get("citation_count"),
    }


def _row_from_inspire_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    metadata = hit.get("metadata")
    if not isinstance(metadata, dict):
        return None
    title = _first_title(metadata)
    if not title:
        return None
    abstract = _abstract_text(metadata)
    record_id = str(hit.get("id") or metadata.get("control_number") or "").strip()
    url = _pick_url(metadata, record_id=record_id or None)
    venue = _venue(metadata) or "INSPIRE-HEP"
    pub_date = _publication_date(metadata)
    citation_count = metadata.get("citation_count")
    snippet_parts = [p for p in (venue, pub_date) if p]
    if isinstance(citation_count, int) and citation_count > 0:
        snippet_parts.append(f"{citation_count} citations")
    snippet = f"{title}. {' — '.join(snippet_parts)}".strip()
    if abstract:
        snippet = abstract[:600]
    preprint = _is_preprint(metadata)
    doi_raw = metadata.get("doi")
    doi = str(doi_raw).strip().lower() if doi_raw else None
    if doi and doi.startswith("http"):
        doi = doi.split("doi.org/", 1)[-1].lower()
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": _authors(metadata),
        "venue": venue,
        "publication_date": pub_date,
        "doi": doi,
        "peer_reviewed": not preprint and bool(_venue(metadata)),
        "preprint": preprint,
        "open_access": True,
        "document_type": "preprint" if preprint else "journal_abstract",
        "inspire_recid": record_id or None,
        "citation_count": citation_count,
    }


def _fetch_inspire_live(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 12.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if not q:
        return {"hits": {"hits": []}}
    try:
        resp = knowledge_get(
            INSPIRE_LITERATURE,
            params={"q": q, "size": max(1, min(max_results, 10))},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("[INSPIRE] search failed: %s", exc)
    return {"hits": {"hits": []}}


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
) -> dict[str, Any]:
    """Load INSPIRE fixture rows or query the public literature API."""
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("inspire_hep_search_ligo.json")
        if fixture is not None:
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
                fixture_query = str(payload.get("query") or "").lower()
                if not fixture_query or fixture_query in q.lower() or q.lower() in fixture_query:
                    return payload
            except Exception as exc:
                logger.warning("[INSPIRE] fixture load failed: %s", exc)
    return _fetch_inspire_live(q, max_results=max_results)


def search_inspire_hep(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search INSPIRE-HEP for open physics literature metadata."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    if isinstance(payload.get("results"), list):
        rows: list[dict[str, Any]] = []
        for entry in payload.get("results") or []:
            if not isinstance(entry, dict):
                continue
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
        row = _row_from_inspire_hit(hit)
        if row:
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
