"""ChEMBL bioactive molecule search adapter (EMBL-EBI REST API)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.ChEMBL")

ADAPTER_ID = "chembl"
RETRIEVAL_METHOD = "chembl_molecule_search"
CHEMBL_MOLECULE_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
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


def _row_from_molecule(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    pref_name = str(item.get("pref_name") or "").strip()
    chembl_id = str(item.get("molecule_chembl_id") or "").strip()
    if not pref_name and not chembl_id:
        return None
    title = pref_name or chembl_id
    props = item.get("molecule_properties") if isinstance(item.get("molecule_properties"), dict) else {}
    structures = item.get("molecule_structures") if isinstance(item.get("molecule_structures"), dict) else {}
    formula = str(props.get("full_molformula") or "").strip()
    weight = str(props.get("full_mwt") or props.get("mw_freebase") or "").strip()
    smiles = str(structures.get("canonical_smiles") or "").strip()
    molecule_type = str(item.get("molecule_type") or "").strip()
    max_phase = str(item.get("max_phase") or "").strip()
    detail_parts = [p for p in (molecule_type, formula, f"MW {weight}" if weight else "", f"phase {max_phase}" if max_phase else "") if p]
    snippet = f"{title}. {'; '.join(detail_parts)}".strip()
    if smiles:
        snippet = f"{snippet} SMILES: {smiles[:120]}".strip()
    url = f"https://www.ebi.ac.uk/chembl/compound_report/{chembl_id}" if chembl_id else "https://www.ebi.ac.uk/chembl/"
    return {
        "title": title,
        "snippet": snippet[:600],
        "full_text": snippet or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("ChEMBL",),
        "venue": "ChEMBL",
        "publication_date": None,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": True,
        "document_type": "bioactive_compound",
        "chembl_id": chembl_id or None,
        "molecular_formula": formula or None,
        "molecular_weight": weight or None,
        "canonical_smiles": smiles or None,
        "max_clinical_phase": max_phase or None,
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
        fixture = _fixture_search_path("chembl_search_aspirin.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[ChEMBL] fixture load failed: %s", exc)

    if not q:
        return {"molecules": []}

    try:
        resp = knowledge_get(
            CHEMBL_MOLECULE_SEARCH,
            params={"q": q, "limit": max(1, min(max_results, 10))},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"molecules": []}
    except BudgetExhaustedError:
        logger.warning("[ChEMBL] budget exhausted; skipping retry")
        return {"molecules": []}
    except Exception as exc:
        logger.warning("[ChEMBL] molecule search failed: %s", exc)
        return {"molecules": []}


def search_chembl(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search ChEMBL bioactive small molecules and drugs."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for item in payload.get("molecules") or []:
        if not isinstance(item, dict):
            continue
        row = _row_from_molecule(item)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
