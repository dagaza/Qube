"""RCSB Protein Data Bank structure search adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get, knowledge_post

logger = logging.getLogger("Qube.Knowledge.PDB")

ADAPTER_ID = "pdb"
RETRIEVAL_METHOD = "pdb_structure_search"
PDB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"


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


def _entry_metadata(pdb_id: str, *, timeout: float) -> dict[str, Any]:
    try:
        resp = knowledge_get(
            f"{PDB_ENTRY}/{pdb_id.upper()}",
            headers=_headers(),
            timeout=timeout,
        )
        if resp.ok:
            payload = resp.json()
            return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug("[PDB] metadata fetch failed for %s: %s", pdb_id, exc)
    return {}


def _row_from_hit(hit: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    pdb_id = str(hit.get("identifier") or metadata.get("rcsb_id") or "").strip().upper()
    if not pdb_id:
        return None
    struct = metadata.get("struct") if isinstance(metadata.get("struct"), dict) else {}
    title = str(struct.get("title") or hit.get("title") or pdb_id).strip()
    entry_info = metadata.get("rcsb_entry_info") if isinstance(metadata.get("rcsb_entry_info"), dict) else {}
    deposit_date = str(entry_info.get("deposit_date") or hit.get("deposit_date") or "")[:10] or None
    snippet = title
    if deposit_date:
        snippet = f"{title} (deposited {deposit_date})"
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": title,
        "url": f"https://www.rcsb.org/structure/{pdb_id}",
        "_adapter": ADAPTER_ID,
        "authors": ("RCSB PDB",),
        "venue": "Protein Data Bank",
        "publication_date": deposit_date,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": True,
        "document_type": "protein_structure",
        "pdb_id": pdb_id,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("pdb_search_hemoglobin.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[PDB] fixture load failed: %s", exc)

    if not q:
        return {"result_set": []}

    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct.title",
                "operator": "contains_phrase",
                "value": q,
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max(1, min(max_results, 10))}},
    }
    try:
        resp = knowledge_post(
            PDB_SEARCH,
            json=payload,
            headers={**_headers(), "Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, dict) else {"result_set": []}
    except BudgetExhaustedError:
        logger.warning("[PDB] budget exhausted; skipping retry")
        return {"result_set": []}
    except Exception as exc:
        logger.warning("[PDB] structure search failed: %s", exc)
        return {"result_set": []}


def search_pdb(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search RCSB PDB macromolecular structures."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for hit in payload.get("result_set") or []:
        if not isinstance(hit, dict):
            continue
        pdb_id = str(hit.get("identifier") or "").strip()
        metadata: dict[str, Any] = {}
        if pdb_id and not (_use_fixtures() and hit.get("title")):
            metadata = _entry_metadata(pdb_id, timeout=12.0)
        elif pdb_id and _use_fixtures():
            metadata = {"struct": {"title": hit.get("title")}, "rcsb_entry_info": {"deposit_date": hit.get("deposit_date")}}
        row = _row_from_hit(hit, metadata)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
