"""arXiv Atom API adapter."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import knowledge_get

logger = logging.getLogger("Qube.Knowledge.arXiv")

ADAPTER_ID = "arxiv"
RETRIEVAL_METHOD = "atom_abstract"
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Search arXiv and return preprint rows with summaries."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    headers = {"User-Agent": USER_AGENT}
    try:
        resp = knowledge_get(
            ARXIV_API,
            params={
                "search_query": f"all:{q}",
                "start": 0,
                "max_results": max(1, min(max_results, 10)),
            },
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[arXiv] search failed: %s", exc)
        return []

    return _parse_arxiv_atom(resp.text)


def _parse_arxiv_atom(xml_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("[arXiv] XML parse failed: %s", exc)
        return rows

    for entry in root.findall("atom:entry", _ATOM_NS):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").split())
        summary = " ".join(
            (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").split()
        )
        if not title and not summary:
            continue
        arxiv_id = _entry_id(entry)
        url = f"https://arxiv.org/abs/{quote(arxiv_id)}" if arxiv_id else None
        authors = tuple(
            (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            for a in entry.findall("atom:author", _ATOM_NS)
        )
        authors = tuple(a for a in authors if a)
        published = (entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or "")[:10]
        doi_el = entry.find("atom:link[@title='doi']", _ATOM_NS)
        doi = None
        if doi_el is not None:
            href = doi_el.get("href") or ""
            if "doi.org/" in href:
                doi = href.split("doi.org/", 1)[-1].strip()
        excerpt = summary[:600] if summary else title
        rows.append(
            {
                "title": title,
                "snippet": excerpt,
                "full_text": summary or None,
                "url": url,
                "_adapter": ADAPTER_ID,
                "authors": authors,
                "venue": "arXiv",
                "publication_date": published or None,
                "doi": doi,
                "peer_reviewed": False,
                "preprint": True,
                "open_access": True,
                "document_type": "preprint",
                "arxiv_id": arxiv_id or None,
            }
        )
    return rows


def _entry_id(entry: ET.Element) -> str:
    raw = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
    if "/abs/" in raw:
        return raw.rsplit("/abs/", 1)[-1]
    return raw.rsplit("/", 1)[-1] if raw else ""
