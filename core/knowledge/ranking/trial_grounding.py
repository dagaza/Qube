"""Trial-acronym grounding boost for scientific evidence ranking (Stage 1 patch)."""

from __future__ import annotations

import re
from typing import Any

from core.knowledge.entities.ids import entity_kind
from core.knowledge.entities.trials import extract_trial_entities
from core.knowledge.ranking.relevance import row_text

_REVIEW_TYPE = re.compile(r"\b(review|meta[- ]analysis|systematic review|comment|editorial)\b", re.I)


def extract_trial_signal(query: str, entity_ids: list[str] | None = None) -> frozenset[str]:
    """
    Return normalized trial slugs when present in query or entity IDs.

    Example outputs: ``emperor-reduced``, ``dapa-hf``.
    """
    signals: set[str] = set()
    for eid, _label in extract_trial_entities(query):
        parts = str(eid).split(":")
        if len(parts) >= 3:
            signals.add(parts[-1])
    for eid in entity_ids or []:
        if entity_kind(str(eid)) == "trial":
            parts = str(eid).split(":")
            if len(parts) >= 3:
                signals.add(parts[-1])
    return frozenset(s for s in signals if s)


def _trial_in_text(signal: str, text: str) -> bool:
    lower = (text or "").lower()
    slug = (signal or "").lower().strip()
    if not slug or not lower:
        return False
    spaced = slug.replace("-", " ")
    if slug in lower or spaced in lower:
        return True
    tokens = [t for t in re.split(r"[\s-]+", spaced) if len(t) >= 3]
    return bool(tokens) and all(t in lower for t in tokens)


def _publication_types(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("publication_types") or ()
    return tuple(str(p).strip() for p in raw if str(p).strip())


def _mesh_terms(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("mesh_terms") or ()
    return tuple(str(m).strip() for m in raw if str(m).strip())


_REAL_WORLD_TITLE = re.compile(
    r"\b(real[- ]life|real[- ]world|observational|practice|registry)\b",
    re.I,
)


def trial_grounding_boost(row: dict[str, Any], trial_signals: frozenset[str]) -> float:
    """Additive boost/penalty when the query targets a named clinical trial."""
    if not trial_signals:
        return 0.0

    title = str(row.get("title") or "")
    body = row_text(row)
    boost = 0.0
    title_hit = any(_trial_in_text(signal, title) for signal in trial_signals)
    body_hit = any(_trial_in_text(signal, body) for signal in trial_signals)

    if title_hit:
        boost += 0.5
    elif body_hit:
        boost += 0.15

    pub_types = " | ".join(_publication_types(row)).lower()
    mesh = " | ".join(_mesh_terms(row)).lower()

    if "randomized controlled trial" in pub_types:
        boost += 0.2
    elif "clinical trial" in pub_types or "clinical trial" in mesh:
        boost += 0.2

    if _REVIEW_TYPE.search(pub_types) or _REVIEW_TYPE.search(title):
        boost -= 0.2

    if _REAL_WORLD_TITLE.search(title):
        boost -= 0.2

    if not title_hit and not body_hit:
        boost -= 0.15

    return boost
