"""IETF RFC adapter — published RFCs via the IETF Datatracker document API.

Plan S22 (D38): the adapter used to send the whole question as one
``title__icontains`` filter to the API index (``/api/v1/doc/``), which has no
documents, so it always returned nothing. It now queries the document
resource for RFCs only. Datatracker filters by substring and cannot order
results, so the adapter asks for the query's most specific term in the title
and the next one in the abstract (title only when that pair finds nothing),
then ranks the documents locally by whole-word matches.
"""

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
DOCS_URL = "https://datatracker.ietf.org/api/v1/doc/document/"
USER_AGENT = "Qube/1.0 (local@qube.app)"
# One page of matches. Filtering on a title and an abstract term usually
# leaves fewer than this; the title-only fallback can leave many more.
FETCH_LIMIT = 50

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./+-]*[A-Za-z0-9]|[A-Za-z0-9]")
# Acronyms and mixed-case protocol names (TLS, QUIC, IPv6, OAuth), optionally
# versioned (HTTP/2).
_ACRONYM_RE = re.compile(r"(?:[A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*|[A-Z]{2,}[0-9]*)(?:/[0-9.]+)?")
_STOPWORDS = frozenset(
    """a an the is are was were be been do does did how what why when where which who whom whose
    of in on for to and or with without by from at as about into over under than that this these
    those it its there their they them can could should would will shall may might must not no yes
    vs versus work works working specify specifies specified define defines defined explain describe
    main use used using between according current latest new standard standards protocol protocols
    rfc rfcs ietf""".split()
)
_STD_LEVELS = {
    "std": "Internet Standard",
    "ds": "Draft Standard",
    "ps": "Proposed Standard",
    "bcp": "Best Current Practice",
    "inf": "Informational",
    "exp": "Experimental",
    "hist": "Historic",
}


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


def query_terms(query: str) -> list[str]:
    """Content terms of ``query`` in order, without stopwords or repeats."""
    terms: list[str] = []
    seen: set[str] = set()
    for tok in _TOKEN_RE.findall(sanitize_api_query(query)):
        low = tok.lower()
        if low in _STOPWORDS or low in seen or (len(tok) < 2 and not tok.isdigit()):
            continue
        seen.add(low)
        terms.append(tok)
    return terms


def search_terms(terms: list[str]) -> tuple[str | None, str | None]:
    """(title term, abstract term): acronyms first, then terms with digits, then the longest."""

    def specificity(term: str) -> tuple[int, int, int]:
        return (
            1 if _ACRONYM_RE.fullmatch(term) else 0,
            1 if any(ch.isdigit() for ch in term) else 0,
            len(term),
        )

    ordered = sorted(terms, key=specificity, reverse=True)
    if not ordered:
        return None, None
    rest = [t for t in ordered[1:] if any(ch.isalpha() for ch in t) or "." in t]
    return ordered[0], (rest[0] if rest else None)


def _has_word(term: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text, re.I) is not None


def rank_documents(objects: list[Any], terms: list[str], title_term: str) -> list[dict[str, Any]]:
    """Documents whose title holds ``title_term`` and that match at least two terms
    (one for a one-term query), best first: a title match counts twice, an abstract
    match once; ties go to the shorter title, then the newer RFC."""
    need = min(2, len(terms))
    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        abstract = str(item.get("abstract") or "")
        if not _has_word(title_term, title):
            continue
        in_title = sum(1 for t in terms if _has_word(t, title))
        in_abstract = sum(1 for t in terms if not _has_word(t, title) and _has_word(t, abstract))
        if in_title + in_abstract < need:
            continue
        number = item.get("rfc_number")
        scored.append((2 * in_title + in_abstract, -len(title), number if isinstance(number, int) else 0, item))
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
    return [s[3] for s in scored]


def _rfc_url(name: str) -> str:
    match = re.search(r"rfc(\d+)", name.lower())
    if match:
        return f"https://www.rfc-editor.org/rfc/rfc{match.group(1)}.html"
    return "https://www.rfc-editor.org/"


def _std_level(item: dict[str, Any]) -> str | None:
    slug = str(item.get("std_level") or "").rstrip("/").rsplit("/", 1)[-1]
    return _STD_LEVELS.get(slug)


def _row_from_doc(item: dict[str, Any]) -> dict[str, Any] | None:
    name = str(item.get("name") or "").strip()
    title = str(item.get("title") or name or "").strip()
    if not title:
        return None
    abstract = " ".join(str(item.get("abstract") or "").split())
    level = _std_level(item)
    label = name.upper() if name else "RFC"
    head = f"{label} ({level}): {title}" if level else f"{label}: {title}"
    snippet = f"{head}. {abstract}" if abstract else head
    url = _rfc_url(name) if name else "https://datatracker.ietf.org/doc/"
    display = f"{name.upper()} — {title}" if name else title
    return {
        "title": display,
        "snippet": snippet[:600],
        "full_text": f"{title}. {abstract}" if abstract else title,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Internet Engineering Task Force",),
        "venue": "IETF",
        # Datatracker's ``time`` is when the record last changed, not when the
        # RFC was published, so no date is given.
        "publication_date": None,
        "document_type": "standard_document",
        "doc_name": name or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _get_documents(params: dict[str, str], *, timeout: float) -> list[Any]:
    resp = knowledge_get(
        DOCS_URL,
        params={**params, "type": "rfc", "limit": FETCH_LIMIT, "format": "json"},
        headers=_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    objects = payload.get("objects") if isinstance(payload, dict) else None
    return objects if isinstance(objects, list) else []


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Datatracker documents for ``search_query`` (unranked), as ``{"objects": [...]}``."""
    del max_results  # Ranking happens after the fetch; the page size is fixed.
    if _use_fixtures():
        fixture = _fixture_search_path("ietf_rfc_search_tls.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[IETF] fixture load failed: %s", exc)

    title_term, abstract_term = search_terms(query_terms(search_query))
    if not title_term:
        return {"objects": []}

    try:
        objects: list[Any] = []
        if abstract_term:
            objects = _get_documents(
                {"title__icontains": title_term, "abstract__icontains": abstract_term}, timeout=timeout
            )
        if not objects:
            objects = _get_documents({"title__icontains": title_term}, timeout=timeout)
        return {"objects": objects}
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
    """Search IETF Datatracker for published RFCs."""
    terms = query_terms(query)
    title_term, _ = search_terms(terms)
    if not title_term or max_results <= 0:
        return []

    payload = fetch_search_results(query, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in rank_documents(payload.get("objects") or [], terms, title_term):
        row = _row_from_doc(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
