"""Condition and syndrome normalizers (HF, STEMI, etc.)."""

from __future__ import annotations

import re

from core.knowledge.entities.ids import make_entity_id

_CONDITION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bheart[\s-]failure\b", re.IGNORECASE),
        "heart_failure",
        "heart failure",
    ),
    (
        re.compile(r"\bhfref\b|\bheart[\s-]failure[\s-]with[\s-]reduced\b", re.IGNORECASE),
        "hfref",
        "HFrEF",
    ),
    (
        re.compile(r"\b(hfpef|heart[\s-]failure[\s-]with[\s-]preserved)\b", re.IGNORECASE),
        "hfpef",
        "HFpEF",
    ),
    (
        re.compile(r"\bstemi\b", re.IGNORECASE),
        "stemi",
        "STEMI",
    ),
    (
        re.compile(r"\bnstemi\b", re.IGNORECASE),
        "nstemi",
        "NSTEMI",
    ),
    (
        re.compile(r"\bmyocardial[\s-]infarction\b", re.IGNORECASE),
        "myocardial_infarction",
        "myocardial infarction",
    ),
)


def extract_condition_entities(text: str) -> tuple[tuple[str, str], ...]:
    found: dict[str, str] = {}
    for pattern, key, label in _CONDITION_PATTERNS:
        if pattern.search(text or ""):
            found[make_entity_id("condition", key)] = label
    return tuple((eid, found[eid]) for eid in sorted(found))
