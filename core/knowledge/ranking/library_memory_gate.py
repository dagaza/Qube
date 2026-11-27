"""Cross-encoder relevance gate for library and memory-recall searches (plan S35).

The S3 cross-encoder decides what plain-chat library search, ``@library`` and
memory recall keep, in place of the embedding floors (D55):

* every candidate the vector and FTS channels returned (the S34 shadow pool) is
  scored against the search text, as the shadow pass scores it: a chunk as
  ``rerank_row_text`` with its file name as the title, a memory as its content
  in the first person (``library_memory_shadow.library_passage`` /
  ``memory_passage``, D56);
* rows scoring below ``keep`` are dropped. They never reach the prompt, the
  sources or the citation numbers; ``@library`` lists them in the retrieval
  inspector (D2);
* survivors are ordered by score, best first, and cut to the search's budget;
* the state is ``sufficient`` with two kept rows ``>= strong``, ``partial``
  with fewer, and ``insufficient`` when candidates were found but none passed
  ``keep``. ``insufficient`` gives the "not found in your documents" wording
  (D57); ``partial`` is recorded only, with no prompt note (D58).

Calibration (S35, V80, V81): on S36's texts, all three embedders, -6 replaces
the floors with 0 rule-7 regressions. Plain-chat library nDCG@5 0.95 / 0.96 /
0.96 (Balanced / Fast / Power; floors 0.90 / 0.86 / 0.80), irrelevant kept
0.39-0.40 (0.73-0.78), no answerable case without a kept row, 10 of 13 abstain
cases with none. With first-person memories, memory recall 0.99 / 0.05 / 1.00 /
1.00 (nDCG@5 / irrelevant kept / recall / abstain correct) on all three.
``strong=2`` is the web's value; D58 keeps it out of the prompt.

Core memory and narrative recaps are not gated (D49, V73): the cross-encoder
judges whether a passage answers the question, and the memories those turns
need do not answer it.

``gate_passages`` returns ``None`` when the reranker cannot run (disabled,
missing, non-English or content-free question, scoring error); callers then
keep today's floors unchanged (fallback-first).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.knowledge.ranking.sufficiency import (
    Sufficiency,
    SufficiencyResult,
    SufficiencyThresholds,
    assess_sufficiency,
)

LIBRARY_THRESHOLDS = SufficiencyThresholds(keep=-6.0, strong=2.0, min_strong_for_sufficient=2)
MEMORY_RECALL_THRESHOLDS = SufficiencyThresholds(keep=-6.0, strong=2.0, min_strong_for_sufficient=2)

STATE_UNSCORED = Sufficiency.UNSCORED.value
STATE_NONE = Sufficiency.NONE.value
STATE_INSUFFICIENT = Sufficiency.INSUFFICIENT.value

_DROPPED_DIAG_LIMIT = 25


@dataclass(frozen=True)
class GateResult:
    scores: tuple[float, ...]  # one per scored passage, input order
    kept_indices: tuple[int, ...]  # passages >= keep, best first (ties keep input order)
    sufficiency: SufficiencyResult  # of the scored pool
    thresholds: SufficiencyThresholds
    latency_ms: float
    unscored: int = 0  # passages past the reranker's row cap (dropped)
    diag: dict[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> str:
        return self.sufficiency.state.value


def gate_passages(
    question: str,
    passages: Sequence[str],
    thresholds: SufficiencyThresholds,
) -> GateResult | None:
    """Score ``passages`` against ``question`` (see module doc); ``None`` = fall back to the floors.

    An empty pool is ``none``, not ``insufficient``: nothing was found to judge.
    """
    from core.knowledge.ranking import rerank

    q = str(question or "").strip()
    if rerank.unavailable_reason(q) is not None:
        return None
    docs = [str(p or "") for p in passages]
    if not docs:
        empty = assess_sufficiency([], thresholds)
        return GateResult((), (), empty, thresholds, 0.0, diag=_diag(empty, thresholds, (), 0.0, 0))
    cap = rerank.DEFAULT_MAX_ROWS
    to_score, unscored = docs[:cap], len(docs) - min(len(docs), cap)
    result = rerank.rerank_texts(q, to_score)
    if result is None:
        return None
    scores = tuple(float(s) for s in result.scores)
    suff = assess_sufficiency(scores, thresholds)
    return GateResult(
        scores=scores,
        kept_indices=suff.kept_indices,
        sufficiency=suff,
        thresholds=thresholds,
        latency_ms=float(result.latency_ms),
        unscored=unscored,
        diag=_diag(suff, thresholds, scores, float(result.latency_ms), unscored),
    )


def _diag(
    suff: SufficiencyResult,
    thresholds: SufficiencyThresholds,
    scores: Sequence[float],
    latency_ms: float,
    unscored: int,
) -> dict[str, Any]:
    """Counts and scores for telemetry and the inspector; no question or passage text."""
    from core.knowledge.ranking import rerank

    ranked = sorted((float(s) for s in scores), reverse=True)
    return {
        "model": rerank.RERANKER_MODEL_ID,
        "mode": "gate",
        "state": suff.state.value,
        "threshold_keep": thresholds.keep,
        "threshold_strong": thresholds.strong,
        "scored": len(scores),
        "unscored": int(unscored),
        "survivors": len(suff.kept_indices),
        "dropped": len(scores) - len(suff.kept_indices) + int(unscored),
        "strong_count": suff.strong_count,
        "top_scores": [round(s, 3) for s in ranked[:5]],
        "latency_ms": round(float(latency_ms), 1),
    }


def kept_diag(diag: dict[str, Any], kept_scores: Sequence[float]) -> dict[str, Any]:
    """``diag`` plus the scores of the rows the search finally kept (after its budget)."""
    out = dict(diag)
    out["kept"] = len(kept_scores)
    out["kept_scores"] = [round(float(s), 3) for s in kept_scores]
    return out


def skipped_diag(question: str) -> dict[str, Any]:
    """Why the gate did not run (telemetry and inspector)."""
    from core.knowledge.ranking import rerank

    try:
        reason = rerank.unavailable_reason(str(question or "").strip()) or rerank.SKIP_ERROR
    except Exception:
        reason = rerank.SKIP_ERROR
    return {"mode": "gate", "state": STATE_UNSCORED, "skipped_reason": reason}


def dropped_for_diag(rows: Sequence[tuple[str, float]]) -> list[dict[str, Any]]:
    """D2: gate-dropped chunks for the retrieval inspector, best first (never the prompt)."""
    ordered = sorted(rows, key=lambda r: -float(r[1]))
    return [
        {"title": str(title or "")[:200], "url": None, "adapter": "lancedb_library", "score": round(float(score), 4)}
        for title, score in ordered[:_DROPPED_DIAG_LIMIT]
    ]


__all__ = [
    "LIBRARY_THRESHOLDS",
    "MEMORY_RECALL_THRESHOLDS",
    "STATE_INSUFFICIENT",
    "STATE_NONE",
    "STATE_UNSCORED",
    "GateResult",
    "dropped_for_diag",
    "gate_passages",
    "kept_diag",
    "skipped_diag",
]
