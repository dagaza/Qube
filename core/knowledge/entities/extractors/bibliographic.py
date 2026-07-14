"""Bibliographic entity extractor (always-on)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.entities.ids import make_entity_id

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_PUBMED_ID_RE = re.compile(r"\bpmid[:\s#]*(\d{6,})\b", re.IGNORECASE)


@dataclass(frozen=True)
class BibliographicExtractor:
    id: str = "bibliographic"
    pack_id: str = "bibliographic"
    kinds: tuple[str, ...] = ("doi", "pubmed")
    priority: int = 0
    cost: str = "cheap"

    def extract(self, text: str, *, doi: str | None = None) -> tuple[str, ...]:
        found: set[str] = set()
        if doi:
            found.add(make_entity_id("doi", doi.lower()))
        for match in _DOI_RE.findall(text or ""):
            found.add(make_entity_id("doi", match.lower()))
        for match in _PUBMED_URL_RE.findall(text or ""):
            found.add(make_entity_id("pubmed", match))
        for match in _PUBMED_ID_RE.findall(text or ""):
            found.add(make_entity_id("pubmed", match))
        return tuple(sorted(found))


BIBLIOGRAPHIC_EXTRACTOR = BibliographicExtractor()
