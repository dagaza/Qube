"""EUR-Lex adapter — EU legal acts via CELLAR SPARQL (no API key required)."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.EurLex")

ADAPTER_ID = "eur_lex"
RETRIEVAL_METHOD = "eur_lex_search"
SPARQL_URL = "https://publications.europa.eu/webapi/rdf/sparql"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"
_ENG_LANG = "http://publications.europa.eu/resource/authority/language/ENG"
_CELEX_RE = re.compile(r"celex:([0-9]{4}[A-Z0-9]+)", re.IGNORECASE)


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
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }


def _sparql_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _celex_from_binding(value: str | None) -> str | None:
    if not value:
        return None
    match = _CELEX_RE.search(str(value))
    return match.group(1).upper() if match else None


def _eurlex_url(celex: str | None) -> str | None:
    if not celex:
        return None
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{quote(celex)}"


def _row_from_binding(binding: dict[str, Any]) -> dict[str, Any] | None:
    title_raw = binding.get("title") or {}
    title = str(title_raw.get("value") or "").strip()
    if not title:
        return None
    celex_doc = binding.get("celexDoc") or {}
    celex = _celex_from_binding(str(celex_doc.get("value") or ""))
    snippet = title[:600]
    return {
        "title": title,
        "snippet": snippet,
        "url": _eurlex_url(celex),
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "EUR-Lex",
        "publication_date": None,
        "document_type": "eu_legal_act",
        "celex": celex,
        "retrieval_method": RETRIEVAL_METHOD,
        "authority_score": 0.94,
        "jurisdiction": "EU",
    }


def _build_sparql_query(search_query: str, *, limit: int) -> str:
    term = _sparql_escape(sanitize_api_query(search_query))
    return f"""
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?title ?celexDoc WHERE {{
  ?work a cdm:work .
  ?expr cdm:expression_belongs_to_work ?work .
  ?expr cdm:expression_uses_language <{_ENG_LANG}> .
  ?expr cdm:expression_title ?title .
  ?work cdm:work_id_document ?celexDoc .
  FILTER(CONTAINS(LCASE(STR(?title)), LCASE("{term}")))
  FILTER(CONTAINS(STR(?celexDoc), "celex:"))
}}
LIMIT {max(1, min(limit, 20))}
""".strip()


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("eur_lex_search_gdpr.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[EUR-Lex] fixture load failed: %s", exc)

    if not q:
        return {"results": {"bindings": []}}

    try:
        resp = knowledge_get(
            SPARQL_URL,
            params={"query": _build_sparql_query(q, limit=max(3, max_results * 3))},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": {"bindings": []}}
    except BudgetExhaustedError:
        logger.warning("[EUR-Lex] budget exhausted; skipping retry")
        return {"results": {"bindings": []}}
    except Exception as exc:
        logger.warning("[EUR-Lex] search failed: %s", exc)
        return {"results": {"bindings": []}}


def search_eur_lex(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search EUR-Lex EU legal acts by English title (CELLAR SPARQL)."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    bindings = (payload.get("results") or {}).get("bindings") or []
    rows: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        row = _row_from_binding(binding)
        if row is None:
            continue
        key = (row.get("celex") or row["title"]).lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
