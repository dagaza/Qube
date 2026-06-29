"""Heuristic scientific discipline detection (Phase 6 Slice 6a)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.adapters.catalog import (
    implemented_adapters_for_ui_group,
)
from core.knowledge.entities.activators.biomedical import BIOMEDICAL_ACTIVATOR
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

SCIENTIFIC_DISCIPLINE_BIOMEDICAL = "biomedical"
SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE = "computer_science"
SCIENTIFIC_DISCIPLINE_ECONOMICS = "economics"
SCIENTIFIC_DISCIPLINE_PHYSICS = "physics"
SCIENTIFIC_DISCIPLINE_GENERAL = "general_science"

DISCIPLINE_UI_GROUP: dict[str, str] = {
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL: "Science",
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE: "Computer Science",
    SCIENTIFIC_DISCIPLINE_ECONOMICS: "Economics",
    SCIENTIFIC_DISCIPLINE_PHYSICS: "Science",
    SCIENTIFIC_DISCIPLINE_GENERAL: "Science",
}

_CS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(machine learning|deep learning|neural network|transformer|"
        r"large language model|llm|nlp|computer vision|algorithm|"
        r"software engineering|compiler|database|gpu|cuda|pytorch|tensorflow|"
        r"reinforcement learning|graph neural|attention mechanism)\b",
        r"\b(cs\.|arxiv:cs|neural machine translation|bert|gpt|diffusion model)\b",
    )
)

_ECON_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(gdp|inflation|econometric|macroeconomic|microeconomic|"
        r"monetary policy|fiscal policy|central bank|interest rate|"
        r"labor market|unemployment|supply.?demand|repec|ssrn)\b",
        r"\b(var model|difference.?in.?differences|panel data regression)\b",
    )
)

_PHYSICS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(gravitational wave|ligo|quantum|particle physics|"
        r"thermodynamic|relativity|astrophys|cosmolog|black hole|"
        r"superconductor|spectroscop)\b",
    )
)

_MEDICAL_HINTS = re.compile(
    r"\b(drug|medication|medicine|disease|symptom|treatment|clinical|patient|"
    r"therapy|diagnosis|fda|vaccine|diabetes|cancer|ozempic|semaglutide)\b",
    re.IGNORECASE,
)


def is_medical_query(query: str) -> bool:
    text = query or ""
    return bool(BIOMEDICAL_ACTIVATOR.matches_query(text) or _MEDICAL_HINTS.search(text))


@dataclass(frozen=True)
class DisciplineMatch:
    discipline: str
    ui_group: str
    scores: dict[str, int]


def _score_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def detect_scientific_discipline(
    query: str,
    *,
    medical_query: bool | None = None,
) -> DisciplineMatch:
    """
    Classify a scholarly query into a discipline bucket for adapter routing.

    Biomedical queries take precedence (PubMed path). Otherwise the highest
    heuristic score wins; ties prefer CS > economics > physics > general.
    """
    text = query or ""
    if medical_query is True or (medical_query is None and is_medical_query(text)):
        discipline = SCIENTIFIC_DISCIPLINE_BIOMEDICAL
        return DisciplineMatch(
            discipline=discipline,
            ui_group=DISCIPLINE_UI_GROUP[discipline],
            scores={discipline: 1},
        )

    scores = {
        SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE: _score_patterns(text, _CS_PATTERNS),
        SCIENTIFIC_DISCIPLINE_ECONOMICS: _score_patterns(text, _ECON_PATTERNS),
        SCIENTIFIC_DISCIPLINE_PHYSICS: _score_patterns(text, _PHYSICS_PATTERNS),
    }
    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        discipline = SCIENTIFIC_DISCIPLINE_GENERAL
    else:
        priority = (
            SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
            SCIENTIFIC_DISCIPLINE_ECONOMICS,
            SCIENTIFIC_DISCIPLINE_PHYSICS,
        )
        discipline = next(d for d in priority if scores[d] == best_score)

    return DisciplineMatch(
        discipline=discipline,
        ui_group=DISCIPLINE_UI_GROUP[discipline],
        scores=scores,
    )


def preferred_adapters_for_discipline(discipline: str) -> tuple[str, ...]:
    """Catalog-defined adapter order for a discipline's UI group (implemented only)."""
    if discipline == SCIENTIFIC_DISCIPLINE_PHYSICS:
        order = implemented_adapters_for_ui_group(
            SERVICE_SCIENTIFIC_EVIDENCE, "Computer Science"
        )
        if order:
            return order
    ui_group = DISCIPLINE_UI_GROUP.get(discipline, "Science")
    return implemented_adapters_for_ui_group(SERVICE_SCIENTIFIC_EVIDENCE, ui_group)
