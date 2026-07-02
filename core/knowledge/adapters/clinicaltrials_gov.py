"""ClinicalTrials.gov adapter — study metadata via public REST API v2."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.ClinicalTrials")

ADAPTER_ID = "clinicaltrials_gov"
RETRIEVAL_METHOD = "clinicaltrials_study_search"
STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
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


def _row_from_study(study: dict[str, Any]) -> dict[str, Any] | None:
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    status_mod = protocol.get("statusModule") or {}
    desc = protocol.get("descriptionModule") or {}
    conditions_mod = protocol.get("conditionsModule") or {}

    nct_id = str(ident.get("nctId") or "").strip()
    title = str(ident.get("briefTitle") or ident.get("officialTitle") or "").strip()
    if not title and not nct_id:
        return None

    summary = str(desc.get("briefSummary") or "").strip()
    status = str(status_mod.get("overallStatus") or "").strip()
    conditions = [
        str(c).strip()
        for c in (conditions_mod.get("conditions") or [])
        if str(c).strip()
    ]
    snippet_parts = [part for part in (status, ", ".join(conditions[:3]), summary[:400]) if part]
    snippet = ". ".join(snippet_parts) if snippet_parts else title
    url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "https://clinicaltrials.gov/"
    display = f"{nct_id} — {title}" if nct_id and title else (title or nct_id)

    return {
        "title": display,
        "snippet": snippet[:600],
        "full_text": summary or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (),
        "venue": "ClinicalTrials.gov",
        "publication_date": None,
        "document_type": "clinical_trial",
        "nct_id": nct_id or None,
        "trial_status": status or None,
        "conditions": tuple(conditions[:5]),
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
        fixture = _fixture_search_path("clinicaltrials_gov_search_diabetes.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[ClinicalTrials] fixture load failed: %s", exc)

    if not q:
        return {"studies": []}

    try:
        resp = knowledge_get(
            STUDIES_URL,
            params={
                "query.term": q,
                "pageSize": max(1, min(max_results, 10)),
                "format": "json",
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"studies": []}
    except BudgetExhaustedError:
        logger.warning("[ClinicalTrials] budget exhausted; skipping retry")
        return {"studies": []}
    except Exception as exc:
        logger.warning("[ClinicalTrials] search failed: %s", exc)
        return {"studies": []}


def search_clinicaltrials_gov(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search ClinicalTrials.gov for registered clinical studies."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for study in payload.get("studies") or []:
        if not isinstance(study, dict):
            continue
        row = _row_from_study(study)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
