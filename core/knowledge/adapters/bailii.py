"""BAILII adapter — UK / Irish case law search (HTML search, no API key)."""

from __future__ import annotations

import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.BAILII")

ADAPTER_ID = "bailii"
RETRIEVAL_METHOD = "bailii_search"
BASE_URL = "https://www.bailii.org"
SEARCH_URL = f"{BASE_URL}/cgi-bin/lucy_search_1.cgi"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"

_RESULT_RE = re.compile(
    r'<a href="(?P<path>/[^"]+\.html)">\s*(?P<title>[^<]+?)\s*</a>\s*'
    r'<i>\(<a href="/cgi-bin/format\.cgi',
    re.IGNORECASE | re.DOTALL,
)
_COURT_SNIPPET_RE = re.compile(
    r"<small>\((?P<snippet>[^<]{0,500}?)</small>",
    re.IGNORECASE | re.DOTALL,
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


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "text/html"}


def _clean_title(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("&amp;", "&")


def _authority_for_path(path: str) -> float:
    lower = path.lower()
    if "/uk/cases/uksc/" in lower or "/ew/cases/uksc/" in lower:
        return 0.95
    if "/ew/cases/ewca/" in lower or "/ew/cases/ewhc/" in lower:
        return 0.88
    if "/eu/cases/echr/" in lower:
        return 0.86
    return 0.82


def _court_from_snippet(snippet: str) -> str | None:
    match = re.search(r"From\s+([^;]+)", snippet)
    if match:
        return match.group(1).strip()
    return None


def _rows_from_html(body: str, *, max_results: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _RESULT_RE.finditer(body):
        path = str(match.group("path") or "").strip()
        if not path or "/cgi-bin/" in path:
            continue
        title = _clean_title(str(match.group("title") or ""))
        if not title:
            continue
        tail = body[match.end() : match.end() + 800]
        snippet_match = _COURT_SNIPPET_RE.search(tail)
        snippet_text = ""
        court = "BAILII"
        if snippet_match:
            snippet_text = _clean_title(snippet_match.group("snippet") or "")
            court = _court_from_snippet(snippet_text) or court
        rows.append(
            {
                "title": title,
                "snippet": (snippet_text or title)[:600],
                "url": urljoin(BASE_URL, path),
                "_adapter": ADAPTER_ID,
                "authors": (),
                "venue": court,
                "publication_date": None,
                "document_type": "court_opinion",
                "citation": [],
                "court": court,
                "retrieval_method": RETRIEVAL_METHOD,
                "authority_score": _authority_for_path(path),
                "jurisdiction": "UK",
            }
        )
        if len(rows) >= max_results:
            break
    return rows


def fetch_search_html(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> str:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("bailii_search_privacy.html")
        if fixture is not None:
            try:
                return fixture.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("[BAILII] fixture load failed: %s", exc)

    if not q:
        return ""

    try:
        resp = knowledge_get(
            SEARCH_URL,
            params={
                "query": q,
                "method": "boolean",
                "mask_path": "",
                "show": max(3, max_results),
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.text
    except BudgetExhaustedError:
        logger.warning("[BAILII] budget exhausted; skipping retry")
        return ""
    except Exception as exc:
        logger.warning("[BAILII] search failed: %s", exc)
        return ""


def search_bailii(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search UK and Irish case law on BAILII."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []
    body = fetch_search_html(q, max_results=max_results)
    if not body:
        return []
    return _rows_from_html(body, max_results=max_results)
