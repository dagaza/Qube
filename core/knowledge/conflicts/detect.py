"""Conflict detection v1 for scientific evidence bundles."""

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


def _stance_label(text: str) -> str:
    t = text or ""
    if _NEGATIVE.search(t):
        return "negative"
    if _POSITIVE.search(t):
        return "positive"
    if _MIXED.search(t):
        return "mixed"
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
        text = f"{src.title}\n{src.excerpt or ''}\n{src.full_text or ''}"
        label = _stance_label(text)
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
