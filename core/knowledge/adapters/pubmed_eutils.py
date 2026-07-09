"""PubMed NCBI E-utilities adapter (search + abstract fetch)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import merge_query_params
from core.knowledge.http_client import knowledge_get

logger = logging.getLogger("Qube.Knowledge.PubMed")

ADAPTER_ID = "pubmed"
RETRIEVAL_METHOD = "eutils_abstract"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
EUTILS_PARAMS = {"tool": "Qube", "email": "local@qube.app"}


def search_pubmed(
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Search PubMed and return abstract rows with bibliographic metadata."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    headers = {"User-Agent": USER_AGENT}
    try:
        search_resp = knowledge_get(
            f"{EUTILS_BASE}esearch.fcgi",
            params=merge_query_params(
                {
                    **EUTILS_PARAMS,
                    "db": "pubmed",
                    "term": q,
                    "retmax": max(1, min(max_results, 10)),
                    "retmode": "json",
                },
                "ncbi",
            ),
            headers=headers,
            timeout=timeout,
        )
        search_resp.raise_for_status()
        pmids = (
            (search_resp.json().get("esearchresult") or {}).get("idlist") or []
        )
    except Exception as exc:
        logger.warning("[PubMed] esearch failed: %s", exc)
        return []

    pmids = [str(p).strip() for p in pmids if str(p).strip()][:max_results]
    if not pmids:
        return []

    try:
        fetch_resp = knowledge_get(
            f"{EUTILS_BASE}efetch.fcgi",
            params=merge_query_params(
                {
                    **EUTILS_PARAMS,
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                },
                "ncbi",
            ),
            headers=headers,
            timeout=timeout,
        )
        fetch_resp.raise_for_status()
    except Exception as exc:
        logger.warning("[PubMed] efetch failed: %s", exc)
        return []

    return _parse_pubmed_xml(fetch_resp.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("[PubMed] XML parse failed: %s", exc)
        return rows

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        pmid = (medline.findtext("PMID") or "").strip()
        art = medline.find("Article")
        if art is None:
            continue
        title = " ".join((art.findtext("ArticleTitle") or "").split())
        abstract = _extract_abstract(art.find("Abstract"))
        if not title and not abstract:
            continue
        journal = art.find("Journal")
        venue = ""
        pub_date = ""
        if journal is not None:
            venue = (journal.findtext("Title") or "").strip()
            pub_date = _format_pub_date(journal.find("JournalIssue/PubDate"))

        authors = tuple(
            _format_author(a)
            for a in art.findall("AuthorList/Author")
            if _format_author(a)
        )
        doi = _extract_doi(article)
        publication_types = _extract_publication_types(art)
        mesh_terms = _extract_mesh_terms(medline)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        excerpt = abstract[:600] if abstract else title
        rows.append(
            {
                "title": title or f"PubMed {pmid}",
                "snippet": excerpt,
                "full_text": abstract or None,
                "url": url,
                "_adapter": ADAPTER_ID,
                "authors": authors,
                "venue": venue or None,
                "publication_date": pub_date or None,
                "doi": doi,
                "peer_reviewed": True,
                "preprint": False,
                "open_access": None,
                "document_type": "journal_abstract",
                "pmid": pmid or None,
                "publication_types": publication_types,
                "mesh_terms": mesh_terms,
            }
        )
    return rows


def _extract_abstract(abstract_el: ET.Element | None) -> str:
    if abstract_el is None:
        return ""
    parts = [t.strip() for t in abstract_el.itertext() if t and t.strip()]
    return " ".join(parts).strip()


def _format_author(author_el: ET.Element) -> str:
    last = (author_el.findtext("LastName") or "").strip()
    fore = (author_el.findtext("ForeName") or author_el.findtext("Initials") or "").strip()
    if last and fore:
        return f"{last}, {fore}"
    return last or fore or (author_el.findtext("CollectiveName") or "").strip()


def _format_pub_date(pub_el: ET.Element | None) -> str:
    if pub_el is None:
        return ""
    year = (pub_el.findtext("Year") or "").strip()
    month = (pub_el.findtext("Month") or "").strip()
    if year and month:
        return f"{year}-{month}"
    return year


def _extract_publication_types(article: ET.Element) -> tuple[str, ...]:
    types = [
        (el.text or "").strip()
        for el in article.findall("PublicationTypeList/PublicationType")
        if (el.text or "").strip()
    ]
    return tuple(dict.fromkeys(types))


def _extract_mesh_terms(medline: ET.Element) -> tuple[str, ...]:
    terms = [
        (el.text or "").strip()
        for el in medline.findall("MeshHeadingList/MeshHeading/DescriptorName")
        if (el.text or "").strip()
    ]
    return tuple(dict.fromkeys(terms))


def _extract_doi(article: ET.Element) -> str | None:
    for id_el in article.findall(".//ArticleId"):
        if (id_el.get("IdType") or "").lower() == "doi":
            doi = (id_el.text or "").strip()
            return doi or None
    return None
