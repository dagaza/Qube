"""Congress.gov adapter — U.S. federal bills via Congress.gov API v3."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.credential_resolver import merge_query_params
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.CongressGov")

ADAPTER_ID = "congress_gov"
RETRIEVAL_METHOD = "congress_gov_bill_search"
CONGRESS_API = "https://api.congress.gov/v3/bill"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_DEFAULT_CONGRESSES = (119, 118)
_BILL_REF_RE = re.compile(
    r"\b(?:(?P<congress>1\d{2})\s*(?:th|st|nd|rd)?\s+)?"
    r"(?P<type>hr|h\.?\s*r\.?|s\.?|hres|h\.?\s*res\.?|sres|s\.?\s*res\.?"
    r"|hconres|sconres|hjres|sjres)\s*[-#.]?\s*(?P<number>\d+)\b",
    re.IGNORECASE,
)
_TYPE_MAP = {
    "hr": "hr",
    "h.r.": "hr",
    "h r": "hr",
    "s": "s",
    "s.": "s",
    "hres": "hres",
    "h.res.": "hres",
    "h res": "hres",
    "sres": "sres",
    "s.res.": "sres",
    "s res": "sres",
    "hconres": "hconres",
    "sconres": "sconres",
    "hjres": "hjres",
    "sjres": "sjres",
}
_CHAMBER_PATH = {
    "hr": "house-bill",
    "s": "senate-bill",
    "hres": "house-resolution",
    "sres": "senate-resolution",
    "hconres": "house-concurrent-resolution",
    "sconres": "senate-concurrent-resolution",
    "hjres": "house-joint-resolution",
    "sjres": "senate-joint-resolution",
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


def _has_api_key() -> bool:
    from core.knowledge.credential_resolver import api_key_query_params

    return bool(api_key_query_params("congress_gov"))


def _normalize_bill_type(raw: str) -> str | None:
    key = re.sub(r"\s+", " ", raw.strip().lower())
    key = key.replace(" ", "")
    if key in _TYPE_MAP:
        return _TYPE_MAP[key]
    compact = raw.strip().lower().replace(".", "").replace(" ", "")
    return _TYPE_MAP.get(compact)


def _parse_bill_reference(query: str) -> tuple[int, str, str] | None:
    match = _BILL_REF_RE.search(query)
    if not match:
        return None
    congress = int(match.group("congress")) if match.group("congress") else 119
    bill_type = _normalize_bill_type(match.group("type") or "")
    number = str(match.group("number") or "").strip()
    if not bill_type or not number:
        return None
    return congress, bill_type, number


def _congress_public_url(congress: int, bill_type: str, number: str) -> str:
    path = _CHAMBER_PATH.get(bill_type.lower(), "house-bill")
    return f"https://www.congress.gov/bill/{congress}th-congress/{path}/{number}"


def _title_matches(title: str, query: str) -> bool:
    tokens = [token for token in re.split(r"\s+", query.lower()) if len(token) >= 3]
    if not tokens:
        return False
    lower = title.lower()
    return all(token in lower for token in tokens)


def _row_from_bill(bill: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(bill, dict):
        return None
    title = str(bill.get("title") or "").strip()
    if not title:
        return None
    congress = bill.get("congress")
    bill_type = str(bill.get("type") or "").strip().lower()
    number = str(bill.get("number") or "").strip()
    latest = bill.get("latestAction") if isinstance(bill.get("latestAction"), dict) else {}
    action_text = str(latest.get("text") or "").strip()
    action_date = str(latest.get("actionDate") or "")[:10] or None
    snippet = action_text[:600] if action_text else title[:600]
    url = _congress_public_url(int(congress), bill_type, number) if congress and bill_type and number else None
    if not url:
        url = str(bill.get("url") or "").strip() or None
    return {
        "title": title,
        "snippet": snippet,
        "full_text": action_text or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (str(bill.get("originChamber") or "U.S. Congress"),),
        "venue": "Congress.gov",
        "publication_date": action_date,
        "document_type": "federal_bill",
        "congress": congress,
        "bill_type": bill_type.upper() if bill_type else None,
        "bill_number": number or None,
        "retrieval_method": RETRIEVAL_METHOD,
        "authority_score": 0.96,
        "jurisdiction": "US",
    }


def _fetch_bill_detail(
    congress: int,
    bill_type: str,
    number: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    url = f"{CONGRESS_API}/{congress}/{bill_type}/{number}"
    resp = knowledge_get(
        url,
        params=merge_query_params({"format": "json"}, "congress_gov"),
        headers=_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    bill = payload.get("bill") if isinstance(payload, dict) else None
    return bill if isinstance(bill, dict) else {}


def _fetch_bill_list(
    congress: int,
    *,
    max_results: int,
    timeout: float,
) -> list[dict[str, Any]]:
    resp = knowledge_get(
        f"{CONGRESS_API}/{congress}",
        params=merge_query_params(
            {"limit": max(25, min(max_results * 8, 100)), "sort": "updateDate+desc"},
            "congress_gov",
        ),
        headers=_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    bills = payload.get("bills") if isinstance(payload, dict) else []
    return [bill for bill in bills or [] if isinstance(bill, dict)]


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("congress_gov_search_privacy.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[CongressGov] fixture load failed: %s", exc)

    if not q or not _has_api_key():
        return {"bills": []}

    try:
        ref = _parse_bill_reference(q)
        if ref is not None:
            congress, bill_type, number = ref
            bill = _fetch_bill_detail(congress, bill_type, number, timeout=timeout)
            return {"bills": [bill]} if bill else {"bills": []}

        matches: list[dict[str, Any]] = []
        for congress in _DEFAULT_CONGRESSES:
            for bill in _fetch_bill_list(congress, max_results=max_results, timeout=timeout):
                title = str(bill.get("title") or "")
                if _title_matches(title, q):
                    matches.append(bill)
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break
        return {"bills": matches}
    except BudgetExhaustedError:
        logger.warning("[CongressGov] budget exhausted; skipping retry")
        return {"bills": []}
    except Exception as exc:
        logger.warning("[CongressGov] bill search failed: %s", exc)
        return {"bills": []}


def search_congress_gov(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search U.S. federal bills on Congress.gov."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for bill in payload.get("bills") or []:
        if not isinstance(bill, dict):
            continue
        row = _row_from_bill(bill)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
