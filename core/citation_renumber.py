"""Renumber citations by first appearance in model output (cited-only sources)."""
from __future__ import annotations

import copy
import re
from typing import Callable

from core.citation_integrity import (
    CITATION_TOKEN_RE,
    normalize_citation_id,
    source_citation_match_keys,
)
from core.citation_normalize import normalize_labeled_citation_tokens


def extract_citation_ids_in_order(text: str) -> list[str]:
    """Citation ids in order of first appearance (after label/combined normalization)."""
    if not text:
        return []
    normalized = normalize_labeled_citation_tokens(text)
    seen: set[str] = set()
    order: list[str] = []
    for match in CITATION_TOKEN_RE.finditer(normalized):
        raw = match.group(1)
        key = "W" if str(raw).lower() == "w" else str(raw)
        if key in seen:
            continue
        seen.add(key)
        order.append(key)
    return order


def _source_for_citation_id(sources: list[dict], cite_id: str) -> dict | None:
    wanted = normalize_citation_id(cite_id)
    if not wanted:
        return None
    for src in sources:
        if not isinstance(src, dict):
            continue
        if wanted in source_citation_match_keys(src):
            return src
    return None


def _token_literal(token_id: str) -> str:
    return "[W]" if str(token_id).upper() == "W" else f"[{token_id}]"


def remap_citation_ids_in_text(text: str, id_map: dict[str, str]) -> str:
    """Replace bracket citation tokens using ``old_id -> new_id`` (values are numeric strings or W)."""
    if not text or not id_map:
        return text or ""

    normalized = normalize_labeled_citation_tokens(text)

    def _repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        key = "W" if str(raw).lower() == "w" else str(raw)
        new_key = id_map.get(key)
        if new_key is None:
            return match.group(0)
        return _token_literal(new_key)

    return CITATION_TOKEN_RE.sub(_repl, normalized)


def renumber_citations_by_appearance(
    text: str,
    sources: list[dict],
    *,
    copy_sources: Callable[[dict], dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Keep only sources cited in ``text``, ordered by first cite in the answer,
    and renumber bracket tokens contiguously from ``[1]``.
    """
    src_list = [s for s in (sources or []) if isinstance(s, dict)]
    if not text or not src_list:
        return text or "", []

    appearance = extract_citation_ids_in_order(text)
    if not appearance:
        return text, []

    copier = copy_sources or (lambda s: copy.deepcopy(s))
    cited_sources: list[dict] = []
    id_map: dict[str, str] = {}

    for old_id in appearance:
        src = _source_for_citation_id(src_list, old_id)
        if src is None:
            continue
        new_id = str(len(cited_sources) + 1)
        id_map[old_id] = new_id
        row = copier(src)
        row["id"] = int(new_id) if new_id.isdigit() else new_id
        cited_sources.append(row)

    if not cited_sources:
        return text, []

    remapped = remap_citation_ids_in_text(text, id_map)
    return remapped, cited_sources
