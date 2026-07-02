"""Heuristic scientific discipline detection (Phase 6 Slice 6a)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.knowledge.adapters.catalog import (
    implemented_adapter_ids,
    implemented_adapters_for_ui_group,
)
from core.knowledge.entities.activators.biomedical import BIOMEDICAL_ACTIVATOR
from core.knowledge.scientific_discipline_packs import (
    DISCIPLINE_PACK_VERSION,
    SCIENTIFIC_DISCIPLINE_BIOLOGY,
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL,
    SCIENTIFIC_DISCIPLINE_CHEMISTRY,
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT,
    SCIENTIFIC_DISCIPLINE_ENGINEERING,
    SCIENTIFIC_DISCIPLINE_GENERAL,
    SCIENTIFIC_DISCIPLINE_MEDICINE,
    SCIENTIFIC_DISCIPLINE_PHYSICS,
    SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE,
    SCIENTIFIC_DISCIPLINE_PSYCHOLOGY,
    SCIENTIFIC_DISCIPLINE_SOCIOLOGY,
    SCIENTIFIC_DISCIPLINE_PACKS,
    get_discipline_pack,
    normalize_discipline_id,
)
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

DISCIPLINE_UI_GROUP: dict[str, str] = {
    pack.id: pack.ui_group for pack in SCIENTIFIC_DISCIPLINE_PACKS
}
DISCIPLINE_UI_GROUP[SCIENTIFIC_DISCIPLINE_BIOMEDICAL] = DISCIPLINE_UI_GROUP[
    SCIENTIFIC_DISCIPLINE_MEDICINE
]

# Re-export pack ids for callers that import from this module.
__all__ = (
    "SCIENTIFIC_DISCIPLINE_BIOLOGY",
    "SCIENTIFIC_DISCIPLINE_BIOMEDICAL",
    "SCIENTIFIC_DISCIPLINE_CHEMISTRY",
    "SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE",
    "SCIENTIFIC_DISCIPLINE_ECONOMICS",
    "SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT",
    "SCIENTIFIC_DISCIPLINE_ENGINEERING",
    "SCIENTIFIC_DISCIPLINE_GENERAL",
    "SCIENTIFIC_DISCIPLINE_PHYSICS",
    "SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE",
    "SCIENTIFIC_DISCIPLINE_PSYCHOLOGY",
    "SCIENTIFIC_DISCIPLINE_SOCIOLOGY",
    "DisciplineMatch",
    "detect_scientific_discipline",
    "discipline_pack_version",
    "is_medical_query",
    "preferred_adapters_for_discipline",
)

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
    r"therapy|diagnosis|fda|vaccine|diabetes|cancer|ozempic|semaglutide|trial|"
    r"hospital|mortality|efficacy|randomized)\b",
    re.IGNORECASE,
)

_BIOLOGY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(crispr|cas9|gene editing|genome|genomic|proteomics|transcriptome|"
        r"rna-seq|single.cell|metagenom|microbiome|phylogen|evolutionary|ecology|"
        r"molecular biology|cell biology|biorxiv|protein folding|"
        r"dna replication|plasmid|knockout mouse|species diversity|synaptic)\b",
        r"\b(sequencing assembly|ortholog|epigenetic|chromatin|ribosome)\b",
    )
)

_CHEMISTRY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(pubchem|smiles|inchi|stoichiometr|synthesis|catalys[ti]s|"
        r"chromatograph|hplc|nmr|mass spectrom|molecular formula|"
        r"molecular weight|acetylsalicylic|cyclooxygenase|cox-2|"
        r"organic chemistry|inorganic chemistry|polymer|electrolyte|"
        r"reaction mechanism|oxidation state|spectroscop|titration|"
        r"ligand binding|enzyme kinetics|active site)\b",
        r"\b(compound|covalent|ionic|solvent|mole fraction|periodic table)\b",
    )
)

_PSYCHOLOGY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(psycholog|psychometric|cognitive psycholog|social psycholog|"
        r"developmental psycholog|experimental psycholog|"
        r"working memory|cognitive load|executive function|"
        r"attention bias|reaction time|stroop|priming effect|"
        r"personality trait|big five|perception|memory recall|learning theory)\b",
        r"\b(behavioral experiment|dual.task|decision making psychology)\b",
    )
)

_SOCIOLOGY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(sociolog|social stratification|social mobility|"
        r"income inequality|gender wage gap|social capital|"
        r"ethnograph|qualitative sociology|demographic trend|"
        r"class consciousness|community survey|social network analysis)\b",
        r"\b(institutional sociology|criminolog|deviance theory)\b",
    )
)

_POLISCI_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(political science|comparative politics|international relations|"
        r"voter turnout|electoral reform|electoral system|"
        r"parliamentary|legislat|democratization|authoritarian|"
        r"geopolit|foreign policy|public opinion poll|"
        r"party affiliation|constitutional design)\b",
        r"\b(presidential election|regime type|state capacity)\b",
    )
)

_EARTH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(climate change|global warming|sea level rise|"
        r"remote sensing|satellite imagery|geospatial|"
        r"ocean temperature|sea surface temperature|"
        r"precipitation anomaly|drought index|"
        r"carbon cycle|greenhouse gas|atmospheric co2|"
        r"noaa dataset|nasa earthdata|cmr collection|"
        r"geoscience|meteorolog|hydrolog|ecosystem model|usgs publication)\b",
        r"\b(ghrsst|modis|sentinel|landsat|reanalysis|gridded climate)\b",
    )
)

_ENGINEERING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(electrical engineering|mechanical engineering|civil engineering|"
        r"embedded systems|signal processing|control systems|"
        r"power electronics|robotics|semiconductor|"
        r"ieee standard|rfc \d+|internet protocol|"
        r"wireless communication|antenna design|vlsi|fpga)\b",
        r"\b(structural engineering|manufacturing process|cybersecurity standard)\b",
    )
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

    Clinical/medicine queries take precedence (PubMed-first path). Life-science
    queries without dominant clinical framing route to biology. Chemistry queries
    route to PubChem when compound/synthesis/spectroscopy signals dominate.
    Social-science queries route to psychology, sociology, or political science.
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
        SCIENTIFIC_DISCIPLINE_BIOLOGY: _score_patterns(text, _BIOLOGY_PATTERNS),
        SCIENTIFIC_DISCIPLINE_CHEMISTRY: _score_patterns(text, _CHEMISTRY_PATTERNS),
        SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE: _score_patterns(text, _CS_PATTERNS),
        SCIENTIFIC_DISCIPLINE_ECONOMICS: _score_patterns(text, _ECON_PATTERNS),
        SCIENTIFIC_DISCIPLINE_PHYSICS: _score_patterns(text, _PHYSICS_PATTERNS),
        SCIENTIFIC_DISCIPLINE_PSYCHOLOGY: _score_patterns(text, _PSYCHOLOGY_PATTERNS),
        SCIENTIFIC_DISCIPLINE_SOCIOLOGY: _score_patterns(text, _SOCIOLOGY_PATTERNS),
        SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE: _score_patterns(
            text, _POLISCI_PATTERNS
        ),
        SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT: _score_patterns(text, _EARTH_PATTERNS),
        SCIENTIFIC_DISCIPLINE_ENGINEERING: _score_patterns(text, _ENGINEERING_PATTERNS),
    }
    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        discipline = SCIENTIFIC_DISCIPLINE_GENERAL
    else:
        priority = (
            SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
            SCIENTIFIC_DISCIPLINE_ENGINEERING,
            SCIENTIFIC_DISCIPLINE_ECONOMICS,
            SCIENTIFIC_DISCIPLINE_EARTH_ENVIRONMENT,
            SCIENTIFIC_DISCIPLINE_PHYSICS,
            SCIENTIFIC_DISCIPLINE_CHEMISTRY,
            SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE,
            SCIENTIFIC_DISCIPLINE_SOCIOLOGY,
            SCIENTIFIC_DISCIPLINE_PSYCHOLOGY,
            SCIENTIFIC_DISCIPLINE_BIOLOGY,
        )
        discipline = next(d for d in priority if scores[d] == best_score)

    return DisciplineMatch(
        discipline=discipline,
        ui_group=DISCIPLINE_UI_GROUP[discipline],
        scores=scores,
    )


def preferred_adapters_for_discipline(discipline: str) -> tuple[str, ...]:
    """Adapter order from discipline pack registry (implemented adapters only)."""
    pack = get_discipline_pack(discipline)
    if pack is not None and pack.status == "active":
        implemented = implemented_adapter_ids(SERVICE_SCIENTIFIC_EVIDENCE)
        ordered = tuple(a for a in pack.resolved_adapter_order() if a in implemented)
        if ordered:
            return ordered
    if normalize_discipline_id(discipline) == SCIENTIFIC_DISCIPLINE_PHYSICS:
        order = implemented_adapters_for_ui_group(
            SERVICE_SCIENTIFIC_EVIDENCE, "Computer Science"
        )
        if order:
            return order
    ui_group = DISCIPLINE_UI_GROUP.get(discipline, "Science")
    return implemented_adapters_for_ui_group(SERVICE_SCIENTIFIC_EVIDENCE, ui_group)


def discipline_pack_version() -> str:
    return DISCIPLINE_PACK_VERSION
