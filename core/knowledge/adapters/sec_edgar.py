"""SEC EDGAR adapter — company resolution + recent filings via data.sec.gov."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from core.knowledge.http_client import knowledge_get

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.SEC")

ADAPTER_ID = "sec_edgar"
RETRIEVAL_METHOD = "sec_submissions"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_TICKER_TOKEN_RE = re.compile(r"\b([A-Z]{1,5})\b")
_FORM_PRIORITY = {
    "10-K": 0,
    "10-K/A": 1,
    "10-Q": 2,
    "10-Q/A": 3,
    "8-K": 4,
    "20-F": 5,
    "6-K": 6,
    "S-1": 7,
    "DEF14A": 8,
}

_tickers_cache: dict[str, Any] | None = None
_tickers_loaded_at: float = 0.0
_TICKERS_TTL_SEC = 86400


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _fixture_tickers_path() -> Path | None:
    path = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "knowledge" / "sec_company_tickers_mini.json"
    return path if path.is_file() else None


def _use_fixtures() -> bool:
    return os.environ.get("QUBE_KNOWLEDGE_FIXTURES", "").strip() == "1"


def load_company_tickers(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return SEC company tickers map (ticker/name → cik metadata)."""
    global _tickers_cache, _tickers_loaded_at
    now = time.time()
    if (
        not force_refresh
        and _tickers_cache is not None
        and (now - _tickers_loaded_at) < _TICKERS_TTL_SEC
    ):
        return _tickers_cache

    fixture = _fixture_tickers_path()
    if _use_fixtures() and fixture is not None and not force_refresh:
        try:
            _tickers_cache = json.loads(fixture.read_text(encoding="utf-8"))
            _tickers_loaded_at = now
            return _tickers_cache
        except Exception as exc:
            logger.warning("[SEC] fixture tickers load failed: %s", exc)

    try:
        resp = knowledge_get(COMPANY_TICKERS_URL, headers=_headers(), timeout=15.0)
        resp.raise_for_status()
        _tickers_cache = resp.json()
        _tickers_loaded_at = now
        return _tickers_cache
    except Exception as exc:
        logger.warning("[SEC] company tickers fetch failed: %s", exc)
        return _tickers_cache or {}


def _iter_companies(tickers_json: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _key, entry in (tickers_json or {}).items():
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        ticker = str(entry.get("ticker") or "").strip().upper()
        cik = entry.get("cik_str") or entry.get("cik")
        if not title or cik is None:
            continue
        rows.append(
            {
                "title": title,
                "ticker": ticker,
                "cik": int(cik),
            }
        )
    return rows


_COMPANY_STOPWORDS = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "company",
        "co",
        "llc",
        "ltd",
        "plc",
        "the",
        "and",
        "for",
        "recent",
        "annual",
        "quarterly",
        "report",
        "reports",
        "revenue",
        "risk",
        "factors",
        "filings",
        "filing",
    }
)


def resolve_company(query: str, tickers_json: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Best-effort company match from query text."""
    q = (query or "").strip()
    if not q:
        return None
    companies = _iter_companies(tickers_json or load_company_tickers())
    if not companies:
        return None

    ticker_map = {row["ticker"]: row for row in companies if row.get("ticker")}
    upper = q.upper()
    for token in _TICKER_TOKEN_RE.findall(upper):
        hit = ticker_map.get(token)
        if hit is not None:
            return dict(hit)

    q_lower = q.lower()
    q_tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", q_lower)
        if len(t) >= 3 and t not in _COMPANY_STOPWORDS
    ]
    best: dict[str, Any] | None = None
    best_score = 0
    for row in companies:
        title_lower = row["title"].lower()
        title_tokens = [
            t
            for t in re.findall(r"[a-z0-9]+", title_lower)
            if len(t) >= 3 and t not in _COMPANY_STOPWORDS
        ]
        overlap = len(set(q_tokens) & set(title_tokens))
        if title_tokens:
            head = title_tokens[0]
            if q_lower.startswith(head):
                overlap += 5
            elif head in q_tokens:
                overlap += 3
        if title_lower.replace(".", "") in q_lower or q_lower in title_lower.replace(".", ""):
            overlap += 4
        if overlap > best_score:
            best_score = overlap
            best = dict(row)
    return best if best_score >= 3 else None


def fetch_submissions(cik: int, *, timeout: float = 12.0) -> dict[str, Any] | None:
    url = SUBMISSIONS_URL.format(cik=int(cik))
    try:
        resp = knowledge_get(url, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.warning("[SEC] submissions fetch failed cik=%s: %s", cik, exc)
        return None


def _filing_url(cik: int, accession: str, primary_doc: str) -> str:
    cik_path = str(int(cik))
    acc_nodash = accession.replace("-", "")
    doc = quote(primary_doc or f"{accession}-index.htm")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_path}/{acc_nodash}/{doc}"


def _form_rank(form: str) -> int:
    normalized = (form or "").upper().replace(" ", "")
    return _FORM_PRIORITY.get(normalized, 50)


def _rows_from_submissions(
    submissions: dict[str, Any],
    *,
    company_name: str,
    cik: int,
    form_filter: tuple[str, ...] = (),
    max_results: int = 3,
) -> list[dict[str, Any]]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    if not forms:
        return []

    allowed = {f.upper().replace(" ", "") for f in form_filter if f}
    rows: list[dict[str, Any]] = []
    n = len(forms)
    for i in range(n):
        form = str(forms[i] or "").strip()
        if not form:
            continue
        form_norm = form.upper().replace(" ", "")
        if allowed and form_norm not in allowed and form.split("/")[0].upper() not in allowed:
            continue
        filing_date = str(recent.get("filingDate", [""])[i] or "").strip()
        report_date = str(recent.get("reportDate", [""])[i] or "").strip()
        accession = str(recent.get("accessionNumber", [""])[i] or "").strip()
        primary = str(recent.get("primaryDocument", [""])[i] or "").strip()
        if not accession:
            continue
        title = f"{form} — {company_name}"
        snippet = (
            f"{form} filed {filing_date or 'unknown date'}"
            f"{f' (report period {report_date})' if report_date else ''} "
            f"for {company_name} (CIK {int(cik)})."
        )
        rows.append(
            {
                "_adapter": ADAPTER_ID,
                "title": title,
                "snippet": snippet,
                "url": _filing_url(cik, accession, primary),
                "document_type": "sec_filing",
                "publication_date": filing_date or None,
                "venue": "SEC EDGAR",
                "form": form,
                "company": company_name,
                "cik": str(int(cik)),
                "accession_number": accession,
                "report_date": report_date or None,
                "_form_rank": _form_rank(form),
            }
        )

    rows.sort(
        key=lambda r: (
            r.get("_form_rank", 99),
            -(int((r.get("publication_date") or "1970-01-01").replace("-", "") or 0)),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen_forms: set[str] = set()
    for row in rows:
        base_form = str(row.get("form") or "").split("/")[0].upper()
        if base_form in seen_forms and len(deduped) >= 1:
            continue
        seen_forms.add(base_form)
        deduped.append(row)
        if len(deduped) >= max_results:
            break
    return deduped[:max_results]


def search_sec_edgar(
    query: str,
    *,
    form_filter: tuple[str, ...] = (),
    max_results: int = 3,
    timeout: float = 12.0,
) -> list[dict[str, Any]]:
    """Resolve a company and return recent SEC filing rows."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    company = resolve_company(q)
    if company is None:
        logger.info("[SEC] no company match for query=%r", q[:120])
        return []

    submissions = fetch_submissions(int(company["cik"]), timeout=timeout)
    if submissions is None:
        return []

    name = str(submissions.get("name") or company.get("title") or "").strip()
    return _rows_from_submissions(
        submissions,
        company_name=name,
        cik=int(company["cik"]),
        form_filter=form_filter,
        max_results=max_results,
    )
