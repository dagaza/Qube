"""Evidence sufficiency from cross-encoder relevance scores (plan S4).

The reranker (``core.knowledge.ranking.rerank``) returns raw MS-MARCO MiniLM
logits. ``assess_sufficiency`` turns them into a decision the pipeline, the
bundle builder and the prompt can share:

* rows scoring below ``keep`` are dropped (never shown to the model);
* ``SUFFICIENT`` — at least ``min_strong_for_sufficient`` kept rows score
  ``>= strong``;
* ``PARTIAL`` — something passed ``keep`` but not enough strong rows;
* ``INSUFFICIENT`` — sources were found, none passed ``keep``;
* ``NONE`` — there was nothing to score;
* ``UNSCORED`` — the reranker did not run (missing, disabled, non-English,
  content-free question); callers keep the legacy ranking unchanged.

Thresholds come from the S2 calibration (``eval/fixtures/knowledge_quality``,
§12 progress log "S4 calibration"): on the 21 scientific fixtures, ``keep=0``
gave abstain precision and recall of 1.00 in both query modes, and no
threshold separated the remaining irrelevant rows (on-subject, wrong facet)
from relevant ones. ``strong=2`` matched the labelled sufficient/partial split
better than 3 or 4. Title-only rows did not justify their own thresholds
(5 useful rows vs 1 irrelevant in the corpus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Sufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    NONE = "none"
    UNSCORED = "unscored"


@dataclass(frozen=True)
class SufficiencyThresholds:
    keep: float
    strong: float
    min_strong_for_sufficient: int = 2


@dataclass(frozen=True)
class SufficiencyResult:
    state: Sufficiency
    # Indices (into the scored sequence) of rows at or above ``keep``, ordered by
    # descending score; ties keep input order.
    kept_indices: tuple[int, ...]
    strong_count: int
    max_score: float | None
    rationale: str


SCIENTIFIC_THRESHOLDS = SufficiencyThresholds(keep=0.0, strong=2.0, min_strong_for_sufficient=2)


def assess_sufficiency(
    scores: Sequence[float] | None,
    thresholds: SufficiencyThresholds,
) -> SufficiencyResult:
    """Classify a scored pool. ``scores is None`` means the reranker did not run."""
    if scores is None:
        return SufficiencyResult(
            state=Sufficiency.UNSCORED,
            kept_indices=(),
            strong_count=0,
            max_score=None,
            rationale="Relevance reranker did not run; legacy ranking applies.",
        )
    values = [float(s) for s in scores]
    if not values:
        return SufficiencyResult(
            state=Sufficiency.NONE,
            kept_indices=(),
            strong_count=0,
            max_score=None,
            rationale="No candidate sources to assess.",
        )
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    kept = tuple(i for i in order if values[i] >= thresholds.keep)
    strong = sum(1 for i in kept if values[i] >= thresholds.strong)
    top = values[order[0]]
    if not kept:
        state = Sufficiency.INSUFFICIENT
        rationale = (
            f"{len(values)} source(s) found; none reached the relevance threshold "
            f"(best {top:.2f} < {thresholds.keep:.2f})."
        )
    elif strong >= max(1, thresholds.min_strong_for_sufficient):
        state = Sufficiency.SUFFICIENT
        rationale = f"{strong} strongly relevant source(s) of {len(kept)} kept."
    else:
        state = Sufficiency.PARTIAL
        rationale = (
            f"{len(kept)} source(s) passed the relevance threshold; "
            f"only {strong} strongly relevant."
        )
    return SufficiencyResult(
        state=state,
        kept_indices=kept,
        strong_count=strong,
        max_score=top,
        rationale=rationale,
    )


def relevance_probability(logit: float) -> float:
    """Map a reranker logit to 0–1 (``EvidenceObject.relevance_score``)."""
    x = float(logit)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
