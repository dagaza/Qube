"""Biomedical condition entity extractor."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.entities.conditions import extract_condition_entities


@dataclass(frozen=True)
class BiomedicalConditionsExtractor:
    id: str = "biomedical_conditions"
    pack_id: str = "biomedical"
    kinds: tuple[str, ...] = ("condition",)
    priority: int = 11
    cost: str = "cheap"

    def extract(self, text: str, *, doi: str | None = None) -> tuple[str, ...]:
        return tuple(eid for eid, _label in extract_condition_entities(text))


BIOMEDICAL_CONDITIONS_EXTRACTOR = BiomedicalConditionsExtractor()
