"""openFDA adapter — drug label and device metadata (FDA)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.OpenFDA")

ADAPTER_ID = "openfda"
RETRIEVAL_METHOD = "openfda_drug_label_search"
DRUG_LABEL_URL = "https://api.fda.gov/drug/label.json"
USER_AGENT = "Qube/1.0 (local@qube.app)"


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


def _fda_search_query(q: str) -> str:
    safe = q.replace('"', "").strip()
    if not safe:
        return ""
    return (
        f'(indications_and_usage:"{safe}" OR purpose:"{safe}" OR '
        f'openfda.brand_name:"{safe}" OR openfda.generic_name:"{safe}")'
    )


def _row_from_result(item: dict[str, Any]) -> dict[str, Any] | None:
    openfda = item.get("openfda") or {}
    brands = openfda.get("brand_name") or []
    generics = openfda.get("generic_name") or []
    title = ""
    if brands:
        title = str(brands[0]).strip()
    elif generics:
        title = str(generics[0]).strip()
    if not title:
        title = "FDA drug label"

    indications = item.get("indications_and_usage") or item.get("purpose") or []
    if isinstance(indications, list):
        indication_text = " ".join(str(x).strip() for x in indications if str(x).strip())
    else:
        indication_text = str(indications).strip()

    warnings = item.get("warnings") or item.get("boxed_warning") or []
    if isinstance(warnings, list):
        warning_text = " ".join(str(x).strip() for x in warnings[:1] if str(x).strip())
    else:
        warning_text = str(warnings).strip()

    snippet_parts = [part for part in (indication_text[:400], warning_text[:200]) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else title

    set_id = str(item.get("set_id") or "").strip()
    url = (
        f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
        if set_id
        else "https://open.fda.gov/"
    )

    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": indication_text or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("U.S. Food and Drug Administration",),
        "venue": "openFDA",
        "publication_date": str(item.get("effective_time") or "")[:8] or None,
        "document_type": "regulatory_label",
        "brand_names": tuple(str(b).strip() for b in brands[:3] if str(b).strip()),
        "generic_names": tuple(str(g).strip() for g in generics[:3] if str(g).strip()),
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 15.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("openfda_search_hypertension.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[OpenFDA] fixture load failed: %s", exc)

    search = _fda_search_query(q)
    if not search:
        return {"results": []}

    try:
        resp = knowledge_get(
            DRUG_LABEL_URL,
            params={"search": search, "limit": max(1, min(max_results, 10))},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": []}
    except BudgetExhaustedError:
        logger.warning("[OpenFDA] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[OpenFDA] search failed: %s", exc)
        return {"results": []}


def search_openfda(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search openFDA drug labels for indications and regulatory text."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_result(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
