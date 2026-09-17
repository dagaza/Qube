"""Conflict detection for evidence bundles."""

from __future__ import annotations

import re

from core.knowledge.types import EvidenceConflict, EvidenceObject

_POSITIVE = re.compile(
    r"\b(effective|benefit|beneficial|improves|reduces risk|safe|successful|"
    r"significant improvement|efficacy)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(ineffective|no benefit|no significant|harmful|adverse|unsafe|"
    r"failed to|does not reduce|worsens)\b",
    re.IGNORECASE,
)
_MIXED = re.compile(
    r"\b(mixed|inconclusive|uncertain|conflicting|limited evidence)\b",
    re.IGNORECASE,
)
_PROSPECTIVE = re.compile(
    r"\b("
    r"scheduled|upcoming|will be held|will tip|set to|expected to|"
    r"to be held|to tip off|tips off|tip off|begins on|starts on|"
    r"kicks off|looking ahead|before the|yet to|has yet to|"
    r"are scheduled|is scheduled|opening game|season opener"
    r")\b",
    re.IGNORECASE,
)
_RETROSPECTIVE_OUTCOME = re.compile(
    r"\b("
    r"won|defeated|beat|beats|clinched|series ended|ended when|"
    r"final score|champions|crowned|was elected|has been elected|"
    r"concluded|wrapped up|are wrapped|completed|finished|"
    r"four games to|game[s]? to one|swept|upset"
    r")\b",
    re.IGNORECASE,
)


def _source_text(src: EvidenceObject) -> str:
    return f"{src.title}\n{src.excerpt or ''}\n{src.full_text or ''}"


def _stance_label(text: str) -> str:
    t = text or ""
    if _NEGATIVE.search(t):
        return "negative"
    if _POSITIVE.search(t):
        return "positive"
    if _MIXED.search(t):
        return "mixed"
    return "neutral"


def _epistemic_label(text: str) -> str:
    t = text or ""
    has_prospective = bool(_PROSPECTIVE.search(t))
    has_retrospective = bool(_RETROSPECTIVE_OUTCOME.search(t))
    if has_prospective and not has_retrospective:
        return "prospective"
    if has_retrospective and not has_prospective:
        return "retrospective_outcome"
    return "neutral"


def detect_conflicts(
    sources: tuple[EvidenceObject, ...],
    *,
    topic: str = "query",
) -> tuple[EvidenceConflict, ...]:
    """Return material conflicts when excerpts cluster into opposing stances."""
    if len(sources) < 2:
        return ()

    clusters: dict[str, list[str]] = {}
    for src in sources:
        label = _stance_label(_source_text(src))
        if label == "neutral":
            continue
        clusters.setdefault(label, []).append(src.title)

    positives = clusters.get("positive") or []
    negatives = clusters.get("negative") or []
    if not positives or not negatives:
        return ()

    positions: list[tuple[str, str]] = []
    if positives:
        positions.append(("supports", positives[0][:120]))
    if negatives:
        positions.append(("contradicts", negatives[0][:120]))
    if len(positions) < 2:
        return ()

    return (
        EvidenceConflict(
            topic=topic[:120],
            positions=tuple(positions),
            severity="material",
        ),
    )


def detect_epistemic_conflicts(
    sources: tuple[EvidenceObject, ...],
    *,
    topic: str = "query",
) -> tuple[EvidenceConflict, ...]:
    """Return conflicts when sources mix prospective and settled-outcome language."""
    if len(sources) < 2:
        return ()

    clusters: dict[str, list[str]] = {}
    for src in sources:
        label = _epistemic_label(_source_text(src))
        if label == "neutral":
            continue
        clusters.setdefault(label, []).append(src.title)

    prospective = clusters.get("prospective") or []
    retrospective = clusters.get("retrospective_outcome") or []
    if not prospective or not retrospective:
        return ()

    return (
        EvidenceConflict(
            topic=topic[:120],
            positions=(
                ("retrospective_outcome", retrospective[0][:120]),
                ("prospective", prospective[0][:120]),
            ),
            severity="material",
        ),
    )


def detect_evidence_conflicts(
    sources: tuple[EvidenceObject, ...],
    *,
    topic: str = "query",
) -> tuple[EvidenceConflict, ...]:
    """Merge stance-based and epistemic conflict detectors."""
    merged: list[EvidenceConflict] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for detector in (detect_conflicts, detect_epistemic_conflicts):
        for conflict in detector(sources, topic=topic):
            key = (conflict.topic, conflict.positions)
            if key in seen:
                continue
            seen.add(key)
            merged.append(conflict)
    return tuple(merged)
