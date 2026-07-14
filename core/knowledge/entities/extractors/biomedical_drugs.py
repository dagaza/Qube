"""Biomedical drug entity extractor."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.entities.drug_classes import extract_drug_entities


@dataclass(frozen=True)
class BiomedicalDrugsExtractor:
    id: str = "biomedical_drugs"
    pack_id: str = "biomedical"
    kinds: tuple[str, ...] = ("drug_class", "drug")
    priority: int = 10
    cost: str = "cheap"

    def extract(self, text: str, *, doi: str | None = None) -> tuple[str, ...]:
        return tuple(eid for eid, _label in extract_drug_entities(text))


BIOMEDICAL_DRUGS_EXTRACTOR = BiomedicalDrugsExtractor()
