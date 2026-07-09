"""Scientific query-type routing (Slice 19).

Classifies @evidence queries into intent buckets (guideline, statistics, standard,
clinical trial, dataset, patent, literature) and reorders enabled adapters so
institutional sources run before bibliographic indexes when appropriate.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

QUERY_TYPE_LITERATURE = "literature"
QUERY_TYPE_GUIDELINE = "guideline"
QUERY_TYPE_STATISTICS = "statistics"
QUERY_TYPE_STANDARD = "standard"
QUERY_TYPE_CLINICAL_TRIAL = "clinical_trial"
QUERY_TYPE_DATASET = "dataset"
QUERY_TYPE_PATENT = "patent"

_ALL_QUERY_TYPES: tuple[str, ...] = (
    QUERY_TYPE_LITERATURE,
    QUERY_TYPE_GUIDELINE,
    QUERY_TYPE_STATISTICS,
    QUERY_TYPE_STANDARD,
    QUERY_TYPE_CLINICAL_TRIAL,
    QUERY_TYPE_DATASET,
    QUERY_TYPE_PATENT,
)

# Institutional adapters eligible for query-type boost (subset of scientific service).
_INSTITUTIONAL_ADAPTER_IDS: frozenset[str] = frozenset(
    {
        "nice",
        "cdc",
        "who",
        "openfda",
        "clinicaltrials_gov",
        "world_bank",
        "eurostat",
        "oecd",
        "bls",
        "us_census",
        "ietf_rfc",
        "nist",
        "ieee_xplore",
        "usgs",
        "noaa",
        "nasa_earthdata",
        "ipcc",
        "copernicus_cds",
        "fao",
        "usda",
        "usda_fdc",
        "uspto_patentsview",
        "epo_espacenet",
        "chembl",
        "uniprot",
        "pdb",
    }
)

_QUERY_TYPE_ADAPTER_ORDER: dict[str, tuple[str, ...]] = {
    QUERY_TYPE_GUIDELINE: (
        "nice",
        "cdc",
        "who",
        "openfda",
        "clinicaltrials_gov",
    ),
    QUERY_TYPE_STATISTICS: (
        "bls",
        "world_bank",
        "eurostat",
        "oecd",
        "us_census",
    ),
    QUERY_TYPE_STANDARD: (
        "ietf_rfc",
        "nist",
        "ieee_xplore",
    ),
    QUERY_TYPE_CLINICAL_TRIAL: (
        "clinicaltrials_gov",
        "openfda",
        "pubmed",
    ),
    QUERY_TYPE_DATASET: (
        "ipcc",
        "copernicus_cds",
        "noaa",
        "nasa_earthdata",
        "usgs",
        "fao",
        "usda",
        "usda_fdc",
    ),
    QUERY_TYPE_PATENT: (
        "uspto_patentsview",
        "epo_espacenet",
    ),
}

_LITERATURE_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(systematic review|meta-analysis|meta analysis|literature review|"
        r"published stud(y|ies)|recent papers?|research papers?|peer.reviewed)\b",
        r"\b(econometric|regression model|VAR model|difference.in.differences|"
        r"instrumental variable|causal inference paper)\b",
        r"\b(mechanism of action|pathway analysis|in vitro|in vivo|"
        r"randomized controlled trial publication)\b",
    )
)

_GUIDELINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bhow (is|are|should) .{0,80} (treated|managed|diagnosed)\b",
        r"\b(treatment|clinical) (guideline|guidelines|recommendation|recommendations)\b",
        r"\bclinical practice guideline\b",
        r"\bstandard of care\b",
        r"\b(nice|cdc|who) (guidance|guideline|recommendation)\b",
        r"\brecommended (treatment|therapy|management) for\b",
        r"\bfirst.line (treatment|therapy)\b",
    )
)

_STATISTICS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(unemployment rate|inflation rate|interest rate|gdp growth rate|"
        r"labor force participation rate|consumer price index|cpi inflation)\b",
        r"\b(current|latest|official) (unemployment|inflation|gdp|cpi)\b",
        r"\bwhat is the (current|latest) .{0,40}(rate|statistics?)\b",
        r"\bofficial statistics\b",
        r"\b(census estimate|census data|population estimate)\b",
        r"\b(eurostat|world bank|bls|oecd) (data|indicator|statistics)\b",
    )
)

_STANDARD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brfc\s*\d+\b",
        r"\b(ietf|ieee) standard\b",
        r"\binternet protocol (standard|specification)\b",
        r"\b(cybersecurity|security) standard\b",
        r"\bnist (standard|guideline|publication)\b",
        r"\bwhat does .{0,40} specify\b",
        r"\b(usb-c|usb c|wifi \d|802\.11)\b.*\b(standard|specification)\b",
    )
)

_CLINICAL_TRIAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bclinicaltrials\.gov\b",
        r"\bclinical trial(s)? (registry|search|recruiting)\b",
        r"\bphase\s*(i{1,3}|[123])\s+(clinical )?trial\b",
        r"\b(randomized|randomised) (controlled )?trial (for|of|comparing)\b",
        r"\brecruiting patients\b",
        r"\btrial registration\b",
    )
)

_DATASET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(climate dataset|satellite dataset|remote sensing dataset|"
        r"gridded climate|reanalysis data|sea surface temperature dataset)\b",
        r"\b(noaa|nasa earthdata|copernicus|usgs) (dataset|data product|catalog)\b",
        r"\b(faostat|food composition database|nutrition dataset)\b",
        r"\b(ipcc|assessment report) (summary|data)\b",
    )
)

_PATENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(patent search|patent application|granted patent|patent portfolio|"
        r"patent landscape|patent filing|patent claim)\b",
        r"\b(uspto|espacenet|patentsview|patent number)\b",
        r"\b(invention disclosure|prior art search)\b",
        r"\b([a-z][\w-]+\s+){1,4}patent\b(?!\s+(ductus|foramen|ovale))",
    )
)

_TYPE_PATTERN_GROUPS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (QUERY_TYPE_GUIDELINE, _GUIDELINE_PATTERNS),
    (QUERY_TYPE_STATISTICS, _STATISTICS_PATTERNS),
    (QUERY_TYPE_STANDARD, _STANDARD_PATTERNS),
    (QUERY_TYPE_CLINICAL_TRIAL, _CLINICAL_TRIAL_PATTERNS),
    (QUERY_TYPE_DATASET, _DATASET_PATTERNS),
    (QUERY_TYPE_PATENT, _PATENT_PATTERNS),
)


@dataclass(frozen=True)
class ScientificQueryTypeMatch:
    """Detected query intent for institutional-first routing."""

    query_type: str
    scores: dict[str, int]


def query_type_routing_enabled() -> bool:
    raw = os.getenv("QUBE_QUERY_TYPE_ROUTING")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _score_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _has_literature_override(text: str) -> bool:
    return any(pattern.search(text) for pattern in _LITERATURE_OVERRIDE_PATTERNS)


def detect_scientific_query_type(query: str) -> ScientificQueryTypeMatch:
    """Score query intent; default to literature when no institutional signal."""
    text = (query or "").strip()
    if not text:
        return ScientificQueryTypeMatch(QUERY_TYPE_LITERATURE, {QUERY_TYPE_LITERATURE: 0})

    if _has_literature_override(text):
        return ScientificQueryTypeMatch(QUERY_TYPE_LITERATURE, {QUERY_TYPE_LITERATURE: 2})

    scores: dict[str, int] = {QUERY_TYPE_LITERATURE: 0}
    for query_type, patterns in _TYPE_PATTERN_GROUPS:
        score = _score_patterns(text, patterns)
        if score:
            scores[query_type] = score

    if len(scores) == 1:
        return ScientificQueryTypeMatch(QUERY_TYPE_LITERATURE, scores)

    ranked = sorted(
        ((qt, sc) for qt, sc in scores.items() if qt != QUERY_TYPE_LITERATURE),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] <= 0:
        return ScientificQueryTypeMatch(QUERY_TYPE_LITERATURE, scores)

    best_type, best_score = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == best_score:
        return ScientificQueryTypeMatch(QUERY_TYPE_LITERATURE, scores)

    return ScientificQueryTypeMatch(best_type, scores)


def institutional_adapters_for_query_type(query_type: str) -> tuple[str, ...]:
    return _QUERY_TYPE_ADAPTER_ORDER.get(query_type, ())


def reorder_adapters_for_query_type(
    adapter_ids: tuple[str, ...],
    *,
    query: str,
    enabled: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Move institutional adapters matching query intent before literature indexes."""
    if not adapter_ids or not query_type_routing_enabled():
        return adapter_ids

    match = detect_scientific_query_type(query)
    if match.query_type == QUERY_TYPE_LITERATURE:
        return adapter_ids

    preferred = institutional_adapters_for_query_type(match.query_type)
    if not preferred:
        return adapter_ids

    enabled_set = set(enabled if enabled is not None else adapter_ids)
    boosted = [aid for aid in preferred if aid in enabled_set and aid in _INSTITUTIONAL_ADAPTER_IDS]
    if not boosted:
        return adapter_ids

    seen: set[str] = set()
    ordered: list[str] = []
    for aid in boosted:
        if aid in seen:
            continue
        ordered.append(aid)
        seen.add(aid)
    for aid in adapter_ids:
        if aid in seen:
            continue
        ordered.append(aid)
        seen.add(aid)
    return tuple(ordered)


def query_type_routing_diag(
    *,
    query: str,
    adapter_ids_before: tuple[str, ...],
    adapter_ids_after: tuple[str, ...],
) -> dict[str, Any]:
    match = detect_scientific_query_type(query)
    return {
        "enabled": query_type_routing_enabled(),
        "query_type": match.query_type,
        "scores": dict(match.scores),
        "adapters_before": list(adapter_ids_before),
        "adapters_after": list(adapter_ids_after),
        "reordered": list(adapter_ids_before) != list(adapter_ids_after),
    }
