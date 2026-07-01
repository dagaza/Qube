"""PubChem adapter — compound records via NCBI PUG REST (fixture + live)."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.PubChem")

ADAPTER_ID = "pubchem"
RETRIEVAL_METHOD = "pubchem_compound"
PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "Qube/1.0 (local assistant; external knowledge platform)"

_QUERY_STOPWORDS = frozenset(
    {
        "binding",
        "bind",
        "bound",
        "enzyme",
        "kinetics",
        "inhibition",
        "inhibitor",
        "cyclooxygenase",
        "cox",
        "cox-2",
        "cox2",
        "receptor",
        "affinity",
        "interaction",
        "mechanism",
        "active",
        "site",
        "substrate",
        "catalysis",
        "molecular",
        "structure",
        "properties",
        "compound",
        "chemical",
        "acid",
    }
)


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


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstract") or entry.get("snippet") or "").strip()
    cid = entry.get("cid") or entry.get("pubchem_cid")
    url = str(entry.get("url") or "").strip() or None
    if not url and cid:
        url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    excerpt = abstract[:600] if abstract else title
    return {
        "title": title,
        "snippet": excerpt,
        "full_text": abstract or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("PubChem",),
        "venue": "PubChem",
        "publication_date": None,
        "doi": entry.get("doi"),
        "peer_reviewed": True,
        "preprint": False,
        "open_access": True,
        "document_type": "compound_record",
        "pubchem_cid": cid,
        "molecular_formula": entry.get("molecular_formula"),
        "molecular_weight": entry.get("molecular_weight"),
        "iupac_name": entry.get("iupac_name"),
    }


def _compound_name_candidates(query: str) -> list[str]:
    q = sanitize_api_query(query)
    if not q:
        return []
    lower = q.lower()
    candidates: list[str] = []
    if "acetylsalicylic acid" in lower:
        candidates.append("acetylsalicylic acid")
    tokens = [t for t in re.split(r"\s+", q) if t]
    for idx in range(len(tokens) - 1):
        bigram = f"{tokens[idx]} {tokens[idx + 1]}"
        if bigram.lower() not in _QUERY_STOPWORDS:
            candidates.append(bigram)
    for token in sorted(tokens, key=len, reverse=True):
        if token.lower() in _QUERY_STOPWORDS:
            continue
        if len(token) >= 4 or token.lower() in {"aspirin", "dmso", "atp", "nad"}:
            candidates.append(token)
    if q not in candidates:
        candidates.append(q)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    return ordered


def _fetch_cid(name: str, *, timeout: float = 10.0) -> int | None:
    encoded = quote(name, safe="")
    url = f"{PUG_BASE}/compound/name/{encoded}/cids/JSON"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        ids = payload.get("IdentifierList", {}).get("CID") or []
        if ids:
            return int(ids[0])
    except Exception as exc:
        logger.debug("[PubChem] CID lookup failed for %r: %s", name, exc)
    return None


def _fetch_compound_record(cid: int, *, timeout: float = 10.0) -> dict[str, Any] | None:
    props_url = (
        f"{PUG_BASE}/compound/cid/{cid}/property/"
        "MolecularFormula,MolecularWeight,IUPACName,Title/JSON"
    )
    desc_url = f"{PUG_BASE}/compound/cid/{cid}/description/JSON"
    headers = {"User-Agent": USER_AGENT}
    try:
        props_resp = requests.get(props_url, headers=headers, timeout=timeout)
        props_resp.raise_for_status()
        props_list = props_resp.json().get("PropertyTable", {}).get("Properties") or []
        props = props_list[0] if props_list else {}
    except Exception as exc:
        logger.warning("[PubChem] property fetch failed for CID %s: %s", cid, exc)
        props = {}

    description = ""
    try:
        desc_resp = requests.get(desc_url, headers=headers, timeout=timeout)
        if desc_resp.ok:
            infos = desc_resp.json().get("Information") or []
            for info in infos:
                if isinstance(info, dict) and info.get("Description"):
                    description = str(info["Description"]).strip()
                    break
    except Exception as exc:
        logger.debug("[PubChem] description fetch failed for CID %s: %s", cid, exc)

    title = str(props.get("Title") or props.get("IUPACName") or f"PubChem CID {cid}").strip()
    iupac = str(props.get("IUPACName") or "").strip()
    formula = str(props.get("MolecularFormula") or "").strip()
    weight = props.get("MolecularWeight")
    parts = [description] if description else []
    if formula:
        parts.append(f"Molecular formula: {formula}")
    if weight is not None:
        parts.append(f"Molecular weight: {weight}")
    if iupac and iupac != title:
        parts.append(f"IUPAC name: {iupac}")
    abstract = "\n".join(parts).strip() or title
    return {
        "title": f"{title} (CID {cid})",
        "abstract": abstract,
        "cid": cid,
        "molecular_formula": formula or None,
        "molecular_weight": weight,
        "iupac_name": iupac or None,
        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
    }


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> dict[str, Any]:
    if _use_fixtures():
        q = sanitize_api_query(search_query).lower()
        for name in (
            "pubchem_search_aspirin.json",
            "pubchem_search_compound.json",
        ):
            fixture = _fixture_search_path(name)
            if fixture is None:
                continue
            try:
                payload = json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[PubChem] fixture load failed: %s", exc)
                continue
            fixture_query = str(payload.get("query") or "").lower()
            if fixture_query and fixture_query not in q and q not in fixture_query:
                continue
            return payload

    rows: list[dict[str, Any]] = []
    for name in _compound_name_candidates(search_query):
        cid = _fetch_cid(name, timeout=timeout)
        if cid is None:
            continue
        record = _fetch_compound_record(cid, timeout=timeout)
        if record is None:
            continue
        record["matched_name"] = name
        rows.append(record)
        if len(rows) >= max(1, max_results):
            break
    return {"results": rows, "source": "pug_rest"}


def search_pubchem(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Resolve compound name(s) to PubChem CID records with properties and description."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    entries = [
        entry for entry in (payload.get("results") or []) if isinstance(entry, dict)
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = _row_from_entry(entry)
        if row.get("title"):
            rows.append(row)
        if len(rows) >= max(1, max_results):
            break
    return rows
