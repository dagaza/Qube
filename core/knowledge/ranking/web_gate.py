"""Cross-encoder relevance gate for web, trusted and finance results (plan S7).

The scientific gate (S4) scores abstracts; this one scores search results
(``title + snippet``, or ``title + full_text`` when a row carries page text) with
the same model and the same decision shape:

* rows below ``keep`` are dropped (shown only in the retrieval inspector, D2);
* survivors are ordered by score, best first;
* ``sufficient`` needs ``min_strong_for_sufficient`` kept rows ``>= strong``.

Thresholds are per service and calibrated on the S2 web / trusted / finance
fixtures (§12 progress log, "S7 calibration"). The calibration landed on the
scientific values (``keep=0``, ``strong=2``, two strong rows), so one score
means the same thing on every path; the constants stay separate so a service
can be re-tuned on its own evidence.

Recency (R9): when the question needs current information
(``core.retrieval_query_cleanup.recency_need``) every row gets a freshness
status (``core.knowledge.ranking.recency``). A row dated outside the window
cannot count as strong; stale rows are ordered after the others; and when no
kept row is fresh but some are stale, the result is at most ``partial`` with
the ``stale_sources`` warning. Undated rows are marked (``undated_sources``
when nothing kept is fresh) but do not lower the state: most snippets carry
no date, and "no date" is not "old".

``gate_rows`` returns ``None`` when the reranker cannot run (missing, disabled,
non-English or content-free question, scoring error); callers then keep their
legacy path unchanged (``UNSCORED``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Sequence
from urllib.parse import urlparse

from core.knowledge.ranking.recency import FRESH, STALE, annotate_freshness
from core.knowledge.ranking.sufficiency import (
    Sufficiency,
    SufficiencyResult,
    SufficiencyThresholds,
    assess_sufficiency,
)
from core.retrieval_query_cleanup import RecencyNeed, recency_need

# S7 calibration on the production-depth pools of the 27 web / trusted / finance
# fixtures, both query modes (54 runs), sweeping keep ∈ {-1, 0, 1, 2},
# strong ∈ {2 … 6}, min ∈ {1, 2}:
# * keep=0 had the lowest irrelevant_kept (0.029) at the best recall (0.889);
#   keep=2 dropped recall to 0.841 and added a false abstention.
# * with min=2, strong=2 matched the labelled state best (0.80; 5 over-, 5
#   under-claims) — strong=4 under-claimed 9 times. The 5 over-claims are pages
#   the labeller marked "partial" that answer the question (e.g. "How to Create
#   a Python Virtual Environment" for "how do I create a venv").
# Trusted (5) and finance (2) fixtures are too few for their own fit.
WEB_THRESHOLDS = SufficiencyThresholds(keep=0.0, strong=2.0, min_strong_for_sufficient=2)
TRUSTED_THRESHOLDS = SufficiencyThresholds(keep=0.0, strong=2.0, min_strong_for_sufficient=2)
FINANCE_THRESHOLDS = SufficiencyThresholds(keep=0.0, strong=2.0, min_strong_for_sufficient=2)

WARNING_RECENCY = "recency_requested"
WARNING_STALE = "stale_sources"
WARNING_UNDATED = "undated_sources"

_DROPPED_DIAG_LIMIT = 25


def is_search_ad_url(url: str) -> bool:
    """DuckDuckGo ad redirects (``duckduckgo.com/y.js?ad_domain=…``) are ads, not results."""
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and parsed.path.startswith("/y.js")


def drop_search_ads(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept = [r for r in rows if not (isinstance(r, dict) and is_search_ad_url(str(r.get("url") or "")))]
    return kept, len(rows) - len(kept)


@dataclass(frozen=True)
class WebGateResult:
    kept: list[dict[str, Any]]  # what to present, best first, at most ``budget``
    survivors: list[dict[str, Any]]  # every row >= keep, in presentation order
    dropped: list[dict[str, Any]]  # rows below keep, best first
    sufficiency: SufficiencyResult  # of ``kept``, recency-adjusted
    warnings: tuple[str, ...] = ()
    recency: RecencyNeed | None = None
    diag: dict[str, Any] = field(default_factory=dict)


def _today(today: date | None) -> date:
    return today or datetime.now(UTC).date()


def review_recency(
    rows: Sequence[dict[str, Any]],
    need: RecencyNeed | None,
    *,
    today: date | None = None,
) -> tuple[str, ...]:
    """Annotate ``rows`` with freshness and return the recency warnings for them."""
    if need is None:
        return ()
    annotate_freshness(rows, window_days=need.window_days, today=_today(today))
    warnings = [WARNING_RECENCY]
    statuses = [r.get("_freshness") for r in rows]
    if rows and FRESH not in statuses:
        warnings.append(WARNING_STALE if STALE in statuses else WARNING_UNDATED)
    return tuple(warnings)


def final_sufficiency(
    kept: Sequence[dict[str, Any]],
    pool_result: SufficiencyResult,
    thresholds: SufficiencyThresholds,
    need: RecencyNeed | None,
) -> SufficiencyResult:
    """Sufficiency of the rows actually presented; stale rows are never strong."""
    if not kept:
        return pool_result
    scores = [r.get("_rerank_score") for r in kept]
    if any(s is None for s in scores):
        return pool_result
    if need is None:
        return assess_sufficiency([float(s) for s in scores], thresholds)
    capped = [
        min(float(s), thresholds.strong - 1e-6) if r.get("_freshness") == STALE else float(s)
        for r, s in zip(kept, scores)
    ]
    result = assess_sufficiency(capped, thresholds)
    statuses = [r.get("_freshness") for r in kept]
    if FRESH not in statuses and STALE in statuses and result.state is Sufficiency.SUFFICIENT:
        result = SufficiencyResult(
            state=Sufficiency.PARTIAL,
            kept_indices=result.kept_indices,
            strong_count=result.strong_count,
            max_score=result.max_score,
            rationale=result.rationale,
        )
    if FRESH not in statuses and STALE in statuses:
        result = SufficiencyResult(
            state=result.state,
            kept_indices=result.kept_indices,
            strong_count=result.strong_count,
            max_score=result.max_score,
            rationale=(
                f"{result.rationale} The question asks for current information; "
                f"no source is dated within the last {need.window_days} days."
            ),
        )
    return result


def gate_rows(
    question: str,
    rows: Sequence[dict[str, Any]],
    *,
    budget: int,
    thresholds: SufficiencyThresholds = WEB_THRESHOLDS,
    today: date | None = None,
    need: RecencyNeed | None | str = "auto",
    max_rows: int | None = None,
) -> WebGateResult | None:
    """Score ``rows`` against ``question`` and select up to ``budget`` (see module doc).

    Rows are annotated in place with ``_rerank_score`` (and freshness fields on
    recency turns). ``need="auto"`` detects the recency need from ``question``.
    """
    from core.knowledge.ranking import rerank

    q = str(question or "").strip()
    if rerank.unavailable_reason(q) is not None:
        return None
    pool = [r for r in rows if isinstance(r, dict)]
    day = _today(today)
    recency = recency_need(q, today=day) if need == "auto" else need
    if not pool:
        empty = assess_sufficiency([], thresholds)
        return WebGateResult(kept=[], survivors=[], dropped=[], sufficiency=empty, recency=recency)
    cap = int(max_rows) if max_rows else rerank.DEFAULT_MAX_ROWS
    to_score, unscored = pool[:cap], pool[cap:]
    result = rerank.rerank_rows(q, to_score, max_rows=cap)
    if result is None:
        return None
    for row, score in zip(to_score, result.scores):
        row["_rerank_score"] = round(float(score), 4)
    pool_result = assess_sufficiency(result.scores, thresholds)
    kept_idx = set(pool_result.kept_indices)
    survivors = [to_score[i] for i in pool_result.kept_indices]
    dropped = sorted(
        (r for i, r in enumerate(to_score) if i not in kept_idx),
        key=lambda r: float(r["_rerank_score"]),
        reverse=True,
    )
    warnings: tuple[str, ...] = ()
    if recency is not None:
        annotate_freshness(survivors, window_days=recency.window_days, today=day)
        # Stable: stale rows go last, otherwise score order is kept.
        survivors.sort(key=lambda r: r.get("_freshness") == STALE)
    kept = survivors[: max(0, int(budget))]
    if recency is not None:
        warnings = review_recency(kept, recency, today=day) if kept else (WARNING_RECENCY,)
    final = final_sufficiency(kept, pool_result, thresholds, recency)
    ranked = sorted((float(s) for s in result.scores), reverse=True)
    diag: dict[str, Any] = {
        "model": rerank.RERANKER_MODEL_ID,
        "mode": "gate",
        "question": q[:300],
        "pool_size": len(pool),
        "threshold_keep": thresholds.keep,
        "threshold_strong": thresholds.strong,
        "latency_ms": round(result.latency_ms, 1),
        "scored": len(result.scores),
        "unscored": len(unscored),
        "top_scores": [round(s, 3) for s in ranked[:5]],
        "kept_scores": [r.get("_rerank_score") for r in kept],
        "survivors": len(survivors),
        "dropped": len(dropped),
        "state": final.state.value,
        "strong_count": final.strong_count,
    }
    if recency is not None:
        diag["recency_window_days"] = recency.window_days
        diag["recency_reason"] = recency.reason
        diag["kept_freshness"] = [r.get("_freshness") for r in kept]
    return WebGateResult(
        kept=kept,
        survivors=survivors,
        dropped=dropped,
        sufficiency=final,
        warnings=warnings,
        recency=recency,
        diag=diag,
    )


def dropped_for_diag(dropped: Sequence[dict[str, Any]], *, adapter: str | None = None) -> list[dict[str, Any]]:
    """D2: gate-dropped rows for the retrieval inspector (never the prompt)."""
    out: list[dict[str, Any]] = []
    for row in list(dropped)[:_DROPPED_DIAG_LIMIT]:
        out.append(
            {
                "title": str(row.get("title") or "")[:200],
                "url": str(row.get("url") or "") or None,
                "adapter": str(row.get("_adapter") or adapter or "") or None,
                "score": row.get("_rerank_score"),
            }
        )
    return out


__all__ = [
    "FINANCE_THRESHOLDS",
    "TRUSTED_THRESHOLDS",
    "WARNING_RECENCY",
    "WARNING_STALE",
    "WARNING_UNDATED",
    "WEB_THRESHOLDS",
    "WebGateResult",
    "drop_search_ads",
    "dropped_for_diag",
    "final_sufficiency",
    "gate_rows",
    "is_search_ad_url",
    "review_recency",
]
