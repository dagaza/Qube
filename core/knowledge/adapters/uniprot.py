"""UniProt protein knowledgebase search adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.UniProt")

ADAPTER_ID = "uniprot"
RETRIEVAL_METHOD = "uniprot_kb_search"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
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


def _protein_name(entry: dict[str, Any]) -> str:
    protein_desc = entry.get("proteinDescription") if isinstance(entry.get("proteinDescription"), dict) else {}
    rec = protein_desc.get("recommendedName") if isinstance(protein_desc.get("recommendedName"), dict) else {}
    full = rec.get("fullName") if isinstance(rec.get("fullName"), dict) else {}
    name = str(full.get("value") or "").strip()
    if name:
        return name
    submitted = protein_desc.get("submissionNames") or []
    if isinstance(submitted, list):
        for item in submitted:
            if not isinstance(item, dict):
                continue
            full_name = item.get("fullName") if isinstance(item.get("fullName"), dict) else {}
            candidate = str(full_name.get("value") or "").strip()
            if candidate:
                return candidate
    return str(entry.get("uniProtkbId") or entry.get("primaryAccession") or "").strip()


def _gene_names(entry: dict[str, Any]) -> tuple[str, ...]:
    genes = entry.get("genes") or []
    if not isinstance(genes, list):
        return ()
    names: list[str] = []
    for gene in genes:
        if not isinstance(gene, dict):
            continue
        gene_name = gene.get("geneName") if isinstance(gene.get("geneName"), dict) else {}
        value = str(gene_name.get("value") or "").strip()
        if value:
            names.append(value)
    return tuple(names)


def _function_text(entry: dict[str, Any]) -> str:
    for comment in entry.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        if str(comment.get("commentType") or "").upper() != "FUNCTION":
            continue
        texts = comment.get("texts") or []
        if isinstance(texts, list):
            for text in texts:
                if isinstance(text, dict):
                    value = str(text.get("value") or "").strip()
                    if value:
                        return value
    return ""


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    accession = str(entry.get("primaryAccession") or "").strip()
    protein = _protein_name(entry)
    if not protein and not accession:
        return None
    organism = entry.get("organism") if isinstance(entry.get("organism"), dict) else {}
    organism_name = str(organism.get("scientificName") or "").strip()
    genes = _gene_names(entry)
    function = _function_text(entry)
    title = protein or accession
    detail_parts = [p for p in (f"gene {', '.join(genes)}" if genes else "", organism_name) if p]
    snippet = function[:600] if function else f"{title}. {'; '.join(detail_parts)}".strip()
    url = f"https://www.uniprot.org/uniprotkb/{accession}" if accession else "https://www.uniprot.org/"
    return {
        "title": title,
        "snippet": snippet[:600] if snippet else title,
        "full_text": function or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": (organism_name,) if organism_name else ("UniProt",),
        "venue": "UniProt",
        "publication_date": None,
        "peer_reviewed": True,
        "preprint": False,
        "open_access": True,
        "document_type": "protein_record",
        "uniprot_accession": accession or None,
        "gene_names": genes,
        "organism": organism_name or None,
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
        fixture = _fixture_search_path("uniprot_search_insulin.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[UniProt] fixture load failed: %s", exc)

    if not q:
        return {"results": []}

    try:
        resp = knowledge_get(
            UNIPROT_SEARCH,
            params={
                "query": q,
                "size": max(1, min(max_results, 10)),
                "format": "json",
            },
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {"results": []}
    except BudgetExhaustedError:
        logger.warning("[UniProt] budget exhausted; skipping retry")
        return {"results": []}
    except Exception as exc:
        logger.warning("[UniProt] search failed: %s", exc)
        return {"results": []}


def search_uniprot(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search UniProt protein records."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    rows: list[dict[str, Any]] = []
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        row = _row_from_entry(entry)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
