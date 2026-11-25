"""Clinical trial acronym normalizers.

Each pattern matches a trial's whole name as words, in queries and in result
rows alike (``trial_name_pattern``). Since plan S23 (D41, V31), the ranking
side no longer matches name parts alone: "dapagliflozin" is not a DAPA-HF hit
and "paradigm" is not a PARADIGM-HF hit. CONSENSUS is the one name that is also
an ordinary word, so it matches in capitals, or as "consensus trial" in any case.
"""

from __future__ import annotations

import re
from functools import lru_cache

from core.knowledge.entities.ids import make_entity_id

# Between name parts: nothing, whitespace, a hyphen, or a Unicode dash.
_SEP = r"[\s\-\u2010-\u2015]?"

_TRIAL_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(rf"\bPARADIGM{_SEP}HF\b", re.I), "paradigm-hf", "PARADIGM-HF"),
    (re.compile(rf"\bEMPEROR{_SEP}Reduced\b", re.I), "emperor-reduced", "EMPEROR-Reduced"),
    (re.compile(rf"\bEMPEROR{_SEP}Preserved\b", re.I), "emperor-preserved", "EMPEROR-Preserved"),
    (re.compile(rf"\bDAPA{_SEP}HF\b", re.I), "dapa-hf", "DAPA-HF"),
    (re.compile(rf"\bDECLARE{_SEP}TIMI{_SEP}58\b", re.I), "declare-timi-58", "DECLARE-TIMI 58"),
    (re.compile(r"\bSOLVD\b", re.I), "solvd", "SOLVD"),
    # Not re.I: "the scientific consensus on …" is not the 1987 trial (V31).
    (re.compile(r"\bCONSENSUS\b|(?i:\bconsensus[\s\-\u2010-\u2015]+trial\b)"), "consensus", "CONSENSUS"),
)
_PATTERN_BY_SLUG = {key: pattern for pattern, key, _label in _TRIAL_PATTERNS}


@lru_cache(maxsize=64)
def trial_name_pattern(slug: str) -> re.Pattern[str] | None:
    """Whole-name pattern for a trial slug such as ``dapa-hf``.

    A slug outside the table (from a resolved entity id) matches all of its
    parts in order, as words, in any case.
    """
    key = (slug or "").strip().lower()
    if key in _PATTERN_BY_SLUG:
        return _PATTERN_BY_SLUG[key]
    parts = [p for p in re.split(r"[\s\-]+", key) if p]
    if not parts:
        return None
    body = _SEP.join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![^\W_]){body}(?![^\W_])", re.I)


def extract_trial_entities(text: str) -> tuple[tuple[str, str], ...]:
    found: dict[str, str] = {}
    for pattern, key, label in _TRIAL_PATTERNS:
        if pattern.search(text or ""):
            found[make_entity_id("trial", key)] = label
    return tuple((eid, found[eid]) for eid in sorted(found))
