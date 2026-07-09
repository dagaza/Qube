"""EPO Espacenet adapter via Open Patent Services (OPS) search API."""

from __future__ import annotations

import base64
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get, knowledge_post

logger = logging.getLogger("Qube.Knowledge.EPO")

ADAPTER_ID = "epo_espacenet"
RETRIEVAL_METHOD = "epo_ops_search"
OPS_TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
OPS_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_EXCH_NS = "http://www.epo.org/exchange"
_OPS_NS = "http://ops.epo.org"


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


def _consumer_credentials() -> tuple[str, str] | None:
    key = os.environ.get("QUBE_EPO_OPS_CONSUMER_KEY", "").strip()
    secret = authorization_token("epo_ops") or ""
    if not key and ":" in secret:
        key, secret = secret.split(":", 1)
    key = key.strip()
    secret = secret.strip()
    if key and secret:
        return key, secret
    return None


def _ops_query(search_query: str) -> str:
    q = sanitize_api_query(search_query)
    if not q:
        return ""
    if "=" in q and any(prefix in q.lower() for prefix in ("ti=", "pa=", "in=", "pn=")):
        return q
    escaped = q.replace('"', '\\"')
    return f'ti="{escaped}"' if " " in q else f"ti={escaped}"


def _fetch_access_token(*, timeout: float) -> str | None:
    creds = _consumer_credentials()
    if creds is None:
        return None
    consumer_key, consumer_secret = creds
    auth = base64.b64encode(f"{consumer_key}:{consumer_secret}".encode()).decode()
    try:
        resp = knowledge_post(
            OPS_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={
                "User-Agent": USER_AGENT,
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = str(payload.get("access_token") or "").strip()
        return token or None
    except Exception as exc:
        logger.warning("[EPO] access token request failed: %s", exc)
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(element: ET.Element, names: tuple[str, ...]) -> str:
    if _local_name(element.tag) in names:
        text = str(element.text or "").strip()
        if text:
            return text
    for child in element.iter():
        if _local_name(child.tag) in names and child.text:
            return str(child.text).strip()
    return ""


def _doc_number_from_exchange(doc: ET.Element) -> str:
    for doc_id in doc.iter():
        if _local_name(doc_id.tag) != "document-id":
            continue
        country = kind = number = ""
        for child in doc_id:
            name = _local_name(child.tag)
            if name == "country":
                country = str(child.text or "").strip()
            elif name == "doc-number":
                number = str(child.text or "").strip()
            elif name == "kind":
                kind = str(child.text or "").strip()
        if number:
            return f"{country}{number}{kind}"
    return ""


def _row_from_exchange_document(doc: ET.Element) -> dict[str, Any] | None:
    title = _find_text(doc, ("invention-title",))
    doc_number = _doc_number_from_exchange(doc)
    if not title and not doc_number:
        return None
    pub_date = _find_text(doc, ("date",))
    pub_date = pub_date[:10] if pub_date else None
    url = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{doc_number}" if doc_number else "https://worldwide.espacenet.com/"
    display_title = title or f"Patent {doc_number}"
    snippet = display_title
    if doc_number:
        snippet = f"{display_title} ({doc_number})"
    return {
        "title": display_title,
        "snippet": snippet[:600],
        "full_text": None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("European Patent Office",),
        "venue": "EPO Espacenet",
        "publication_date": pub_date,
        "document_type": "patent",
        "patent_number": doc_number or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _rows_from_ops_xml(body: str, *, max_results: int) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        logger.warning("[EPO] OPS XML parse failed: %s", exc)
        return []
    rows: list[dict[str, Any]] = []
    for doc in root.iter():
        if _local_name(doc.tag) != "exchange-document":
            continue
        row = _row_from_exchange_document(doc)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("epo_espacenet_search_battery.xml")
        if fixture is not None:
            try:
                return {"xml": fixture.read_text(encoding="utf-8")}
            except Exception as exc:
                logger.warning("[EPO] fixture load failed: %s", exc)

    ops_q = _ops_query(q)
    if not ops_q:
        return {"xml": ""}

    token = _fetch_access_token(timeout=timeout)
    if not token:
        return {"xml": ""}

    try:
        resp = knowledge_get(
            OPS_SEARCH_URL,
            params={"q": ops_q, "Range": f"1-{max(1, min(max_results, 10))}"},
            headers={
                "User-Agent": USER_AGENT,
                "Authorization": f"Bearer {token}",
                "Accept": "application/exchange+xml",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return {"xml": resp.text}
    except BudgetExhaustedError:
        logger.warning("[EPO] budget exhausted; skipping retry")
        return {"xml": ""}
    except Exception as exc:
        logger.warning("[EPO] patent search failed: %s", exc)
        return {"xml": ""}


def search_epo_espacenet(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search worldwide patent bibliographic data via EPO OPS."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    body = str(payload.get("xml") or "")
    if not body:
        return []
    return _rows_from_ops_xml(body, max_results=max_results)
