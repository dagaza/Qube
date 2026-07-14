"""OECD adapter — SDMX dataflow discovery for official statistics."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.OECD")

ADAPTER_ID = "oecd"
RETRIEVAL_METHOD = "oecd_dataflow_search"
DATAFLOW_URL = "https://sdmx.oecd.org/public/rest/dataflow/all"
USER_AGENT = "Qube/1.0 (local@qube.app)"
_TOKEN_SPLIT = re.compile(r"\s+")
_MAX_SCAN = 12_000


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
    return {"User-Agent": USER_AGENT, "Accept": "application/xml"}


def _query_terms(search_query: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in _TOKEN_SPLIT.split(sanitize_api_query(search_query).lower())
        if len(term) >= 3
    )


def _dataflow_score(
    *,
    agency_id: str,
    dataflow_id: str,
    name: str,
    description: str,
    terms: tuple[str, ...],
) -> float:
    if not terms:
        return 0.0
    haystack = f"{agency_id} {dataflow_id} {name} {description}".lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _row_from_dataflow(item: dict[str, Any]) -> dict[str, Any] | None:
    agency_id = str(item.get("agency_id") or "").strip()
    dataflow_id = str(item.get("dataflow_id") or "").strip()
    name = str(item.get("name") or dataflow_id or "").strip()
    if not name:
        return None
    description = str(item.get("description") or "").strip()
    snippet = description[:600] if description else name
    dataflow_ref = f"{agency_id},{dataflow_id}" if agency_id and dataflow_id else dataflow_id
    url = (
        f"https://data-explorer.oecd.org/vis?df[ds]=DisseminateFinalDMZ&df[id]={dataflow_id}&df[ag]=OECD"
        if dataflow_id
        else "https://data.oecd.org/"
    )
    display = f"{dataflow_id} — {name}" if dataflow_id else name
    return {
        "title": display,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Organisation for Economic Co-operation and Development",),
        "venue": "OECD Data Explorer",
        "publication_date": None,
        "document_type": "statistical_release",
        "dataflow_id": dataflow_ref or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _parse_dataflow_xml(content: bytes, *, terms: tuple[str, ...], max_results: int) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    scanned = 0
    for _event, elem in ET.iterparse(io.BytesIO(content), events=("end",)):
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag != "Dataflow":
            continue
        agency_id = str(elem.get("agencyID") or "").strip()
        dataflow_id = str(elem.get("id") or "").strip()
        name = ""
        description = ""
        for child in elem:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag == "Name" and not name:
                name = (child.text or "").strip()
            elif child_tag == "Description" and not description:
                description = (child.text or "").strip()
        score = _dataflow_score(
            agency_id=agency_id,
            dataflow_id=dataflow_id,
            name=name,
            description=description,
            terms=terms,
        )
        if score > 0:
            ranked.append(
                (
                    score,
                    {
                        "agency_id": agency_id,
                        "dataflow_id": dataflow_id,
                        "name": name,
                        "description": description,
                    },
                )
            )
        elem.clear()
        scanned += 1
        if scanned >= _MAX_SCAN and len(ranked) >= max_results:
            break
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in ranked[: max(1, max_results)]]


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 45.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("oecd_search_unemployment.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[OECD] fixture load failed: %s", exc)

    terms = _query_terms(q)
    if not terms:
        return {"dataflows": []}

    try:
        resp = knowledge_get(
            DATAFLOW_URL,
            params={"references": "none", "detail": "allstubs"},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        dataflows = _parse_dataflow_xml(resp.content, terms=terms, max_results=max_results)
        return {"dataflows": dataflows}
    except BudgetExhaustedError:
        logger.warning("[OECD] budget exhausted; skipping retry")
        return {"dataflows": []}
    except Exception as exc:
        logger.warning("[OECD] dataflow search failed: %s", exc)
        return {"dataflows": []}


def search_oecd(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search OECD SDMX dataflows by keyword."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("dataflows") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_dataflow(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
