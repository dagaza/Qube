"""Drug-class normalizers (ACEi, ARB, SGLT2, statins)."""

from __future__ import annotations

import re

from core.knowledge.entities.ids import make_entity_id

# (compiled pattern, entity key, human label)
_DRUG_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"\b(ace[\s-]?inhibitors?|acei|aceis|angiotensin[\s-]converting[\s-]enzyme)\b",
            re.IGNORECASE,
        ),
        "ace_inhibitors",
        "ACE inhibitors",
    ),
    (
        re.compile(
            r"\b(arbs?|angiotensin[\s-]receptor[\s-]blockers?)\b",
            re.IGNORECASE,
        ),
        "arb",
        "ARB",
    ),
    (
        re.compile(
            r"\b(sglt2[\s-]?inhibitors?|sglt[\s-]?2|gliflozins?)\b",
            re.IGNORECASE,
        ),
        "sglt2_inhibitors",
        "SGLT2 inhibitors",
    ),
    (
        re.compile(r"\b(statins?|hmg[\s-]?coa[\s-]?reductase)\b", re.IGNORECASE),
        "statins",
        "Statins",
    ),
)

_DRUG_NAMES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\blisinopril\b", re.I), "lisinopril", "lisinopril"),
    (re.compile(r"\benalapril\b", re.I), "enalapril", "enalapril"),
    (re.compile(r"\bramipril\b", re.I), "ramipril", "ramipril"),
    (re.compile(r"\bempagliflozin\b", re.I), "empagliflozin", "empagliflozin"),
    (re.compile(r"\bdapagliflozin\b", re.I), "dapagliflozin", "dapagliflozin"),
    (re.compile(r"\bcanagliflozin\b", re.I), "canagliflozin", "canagliflozin"),
    (re.compile(r"\bsemaglutide\b", re.I), "semaglutide", "semaglutide"),
    (re.compile(r"\batorvastatin\b", re.I), "atorvastatin", "atorvastatin"),
    (re.compile(r"\brosuvastatin\b", re.I), "rosuvastatin", "rosuvastatin"),
)


def extract_drug_entities(text: str) -> tuple[tuple[str, str], ...]:
    """Return (entity_id, label) pairs found in text."""
    found: dict[str, str] = {}
    for pattern, key, label in _DRUG_PATTERNS:
        if pattern.search(text or ""):
            found[make_entity_id("drug_class", key)] = label
    for pattern, key, label in _DRUG_NAMES:
        if pattern.search(text or ""):
            found[make_entity_id("drug", key)] = label
    return tuple((eid, found[eid]) for eid in sorted(found))
