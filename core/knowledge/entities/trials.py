"""Clinical trial acronym normalizers."""

from __future__ import annotations

import re

from core.knowledge.entities.ids import make_entity_id

_TRIAL_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bPARADIGM[\s-]?HF\b", re.I), "paradigm-hf", "PARADIGM-HF"),
    (re.compile(r"\bEMPEROR[\s-]?Reduced\b", re.I), "emperor-reduced", "EMPEROR-Reduced"),
    (re.compile(r"\bEMPEROR[\s-]?Preserved\b", re.I), "emperor-preserved", "EMPEROR-Preserved"),
    (re.compile(r"\bDAPA[\s-]?HF\b", re.I), "dapa-hf", "DAPA-HF"),
    (re.compile(r"\bDECLARE[\s-]?TIMI[\s-]?58\b", re.I), "declare-timi-58", "DECLARE-TIMI 58"),
    (re.compile(r"\bSOLVD\b", re.I), "solvd", "SOLVD"),
    (re.compile(r"\bCONSENSUS\b", re.I), "consensus", "CONSENSUS"),
)


def extract_trial_entities(text: str) -> tuple[tuple[str, str], ...]:
    found: dict[str, str] = {}
    for pattern, key, label in _TRIAL_PATTERNS:
        if pattern.search(text or ""):
            found[make_entity_id("trial", key)] = label
    return tuple((eid, found[eid]) for eid in sorted(found))
