"""Biomedical clinical trial entity extractor."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.entities.trials import extract_trial_entities


@dataclass(frozen=True)
class BiomedicalTrialsExtractor:
    id: str = "biomedical_trials"
    pack_id: str = "biomedical"
    kinds: tuple[str, ...] = ("trial",)
    priority: int = 12
    cost: str = "cheap"

    def extract(self, text: str, *, doi: str | None = None) -> tuple[str, ...]:
        return tuple(eid for eid, _label in extract_trial_entities(text))


BIOMEDICAL_TRIALS_EXTRACTOR = BiomedicalTrialsExtractor()
