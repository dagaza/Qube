"""OpenAlex works search adapter."""

from __future__ import annotations

import logging
from typing import Any

import requests

from core.knowledge.adapters.query_sanitize import sanitize_api_query

ADAPTER_ID = "openalex"
RETRIEVAL_METHOD = "works_search"
OPENALEX_WORKS = "https://api.openalex.org/works"
USER_AGENT = "Qube/1.0 (mailto:local@qube.app)"

logger = logging.getLogger("Qube.Knowledge.OpenAlex")


def search_openalex(
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Search OpenAlex and return work rows with reconstructed abstracts."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(
            OPENALEX_WORKS,
            params={
                "search": q,
                "per_page": max(1, min(max_results, 10)),
            },
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        results = (resp.json().get("results") or [])[:max_results]
    except Exception as exc:
        logger.warning("[OpenAlex] search failed: %s", exc)
        return []

    rows: list[dict[str, Any]] = []
    for work in results:
        if not isinstance(work, dict):
            continue
        title = str(work.get("display_name") or work.get("title") or "").strip()
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        if not title and not abstract:
            continue
        doi_raw = work.get("doi")
        doi = str(doi_raw).replace("https://doi.org/", "").strip() if doi_raw else None
        url = str(work.get("id") or work.get("primary_location", {}).get("landing_page_url") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        authors = tuple(
            str((a.get("author") or {}).get("display_name") or "").strip()
            for a in (work.get("authorships") or [])
            if isinstance(a, dict)
        )
        authors = tuple(a for a in authors if a)
        venue = str((work.get("primary_location") or {}).get("source", {}).get("display_name") or "").strip()
        pub_year = work.get("publication_year")
        pub_date = str(pub_year) if pub_year else None
        oa = work.get("open_access") if isinstance(work.get("open_access"), dict) else {}
        excerpt = abstract[:600] if abstract else title
        rows.append(
            {
                "title": title,
                "snippet": excerpt,
                "full_text": abstract or None,
                "url": url or None,
                "_adapter": ADAPTER_ID,
                "authors": authors,
                "venue": venue or None,
                "publication_date": pub_date,
                "doi": doi,
                "peer_reviewed": True,
                "preprint": False,
                "open_access": bool(oa.get("is_oa")) if oa else None,
                "document_type": "journal_abstract",
            }
        )
    return rows


def _reconstruct_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        if not positions:
            continue
        max_pos = max(max_pos, max(int(p) for p in positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            idx = int(pos)
            if 0 <= idx < len(words):
                words[idx] = str(word)
    return " ".join(w for w in words if w).strip()
