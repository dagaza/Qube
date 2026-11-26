"""Cross-encoder shadow scores for library and memory-recall searches (plan S34).

The S3 cross-encoder scores the candidates of plain-chat library search,
``@library`` and memory recall, without changing what any search keeps:

* ``LLMWorker.run`` opens a collection for the turn (``begin_turn``), only
  when the reranker is enabled and on disk.
* ``rag_search`` and ``memory_search`` called with a ``shadow_path`` offer
  their candidate pool (every row their vector and FTS channels returned,
  before the relevance floors) and mark the rows they kept (``offer``).
* After the reply, ``end_turn`` hands the jobs to a daemon thread
  (``score_jobs_async``) that scores them and logs one ``[RerankShadow]
  {json}`` INFO line per search. The line holds scores and counts only:
  no question, passage text or file name.

Core memory and narrative recaps get no shadow (D49): the cross-encoder judges
whether a passage answers the question, and the memories those turns need
(a diet preference for "suggest a dinner", session summaries for "what have
we been working on?") score like unrelated rows (V73).

Passages are built the way a gate would read them: a chunk as
``rerank_row_text`` with its file name as the title, a memory as its content.
The thresholds below come from the S34 calibration (V72) and only label the
counts in the log; S35 decides the gate.

Never raises. A search running outside a worker turn offers nothing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from core.knowledge.ranking import rerank

logger = logging.getLogger("Qube.Knowledge.RerankShadow")

PATH_LIBRARY_CHAT = "library_chat"  # plain-chat RAG route
PATH_LIBRARY_TOOL = "library_tool"  # @library
PATH_MEMORY_RECALL = "memory_recall"  # MEMORY / HYBRID route, not narrative
SHADOW_PATHS = (PATH_LIBRARY_CHAT, PATH_LIBRARY_TOOL, PATH_MEMORY_RECALL)

# S34 calibration on the S33 corpus (V72), with the cross-encoder in place of
# the floor: -4 and -6 are the two candidates for chunks (file name in the
# passage) and recall memories. -4 keeps fewer irrelevant rows, -6 a little
# more nDCG. ``would_keep`` / ``would_abstain`` use -4; the counts at -6 and 0
# are logged too. S35 chooses.
CALIBRATED_KEEP: dict[str, float] = {
    PATH_LIBRARY_CHAT: -4.0,
    PATH_LIBRARY_TOOL: -4.0,
    PATH_MEMORY_RECALL: -4.0,
}
SHADOW_THRESHOLDS: tuple[float, ...] = (-6.0, -4.0, 0.0)
MAX_CANDIDATES = rerank.DEFAULT_MAX_ROWS


@dataclass(frozen=True)
class ShadowCandidate:
    passage: str
    kept: bool


@dataclass(frozen=True)
class ShadowJob:
    path: str
    question: str
    top_k: int
    scoped: bool
    candidates: tuple[ShadowCandidate, ...]


_lock = threading.Lock()
_active = False
_jobs: list[ShadowJob] = []


def library_passage(source: str, text: str) -> str:
    """A chunk as a gate would score it: file name as the title, then the chunk."""
    return rerank.rerank_row_text({"title": str(source or ""), "snippet": str(text or "")})


def memory_passage(content: str) -> str:
    return " ".join(str(content or "").split())[: rerank.DEFAULT_TEXT_CHARS]


def begin_turn() -> bool:
    """Start collecting for a turn; ``False`` (and no collection) when the reranker is unavailable."""
    global _active
    try:
        available = rerank.reranker_available()
    except Exception:
        available = False
    with _lock:
        _jobs.clear()
        _active = bool(available)
        return _active


def collecting() -> bool:
    return _active


def offer(
    path: str,
    question: str,
    candidates: Iterable[tuple[str, bool]],
    *,
    top_k: int,
    scoped: bool = False,
) -> None:
    """Record one search's candidates as ``(passage, kept)`` pairs. No-op outside a turn."""
    if not _active or path not in SHADOW_PATHS:
        return
    try:
        rows = [ShadowCandidate(str(p), bool(k)) for p, k in candidates if str(p or "").strip()]
        if not rows:
            return
        # Kept rows first (callers list them in rank order), so the cap never
        # leaves a kept row unscored and ``kept_scores`` follow the ranking.
        rows = [r for r in rows if r.kept] + [r for r in rows if not r.kept]
        job = ShadowJob(path, str(question or ""), int(top_k), bool(scoped), tuple(rows[:MAX_CANDIDATES]))
    except Exception:
        logger.debug("[RerankShadow] offer failed", exc_info=True)
        return
    with _lock:
        if _active:
            _jobs.append(job)


def end_turn() -> list[ShadowJob]:
    """Stop collecting and return the turn's jobs."""
    global _active
    with _lock:
        jobs = list(_jobs)
        _jobs.clear()
        _active = False
    return jobs


def _threshold_key(t: float) -> str:
    return f"{t:g}"


def score_job(job: ShadowJob) -> dict:
    """Score one job; the record logged for it. Never raises."""
    record: dict = {
        "path": job.path,
        "model": rerank.RERANKER_MODEL_ID,
        "candidates": len(job.candidates),
        "kept": sum(1 for c in job.candidates if c.kept),
        "top_k": job.top_k,
        "scoped": job.scoped,
        "skipped_reason": None,
    }
    try:
        reason = rerank.unavailable_reason(job.question)
        if reason is not None:
            record["skipped_reason"] = reason
            return record
        started = time.perf_counter()
        result = rerank.rerank_texts(job.question, [c.passage for c in job.candidates])
        if result is None:
            record["skipped_reason"] = rerank.SKIP_ERROR
            return record
        scores = result.scores
        kept_scores = [s for s, c in zip(scores, job.candidates) if c.kept]
        keep = CALIBRATED_KEEP[job.path]
        above_keep = sum(1 for s in scores if s >= keep)
        record.update(
            {
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "threads": rerank.reranker_threads(),
                "kept_scores": [round(s, 2) for s in kept_scores],
                "pool_max": round(max(scores), 2),
                "pool_at_or_above": {_threshold_key(t): sum(1 for s in scores if s >= t) for t in SHADOW_THRESHOLDS},
                "kept_below": {_threshold_key(t): sum(1 for s in kept_scores if s < t) for t in SHADOW_THRESHOLDS},
                "calibrated_keep": keep,
                "would_keep": min(job.top_k, above_keep),
                "would_abstain": above_keep == 0,
            }
        )
    except Exception as exc:
        record["skipped_reason"] = rerank.SKIP_ERROR
        logger.debug("[RerankShadow] scoring failed: %s", exc)
    return record


def score_jobs(jobs: Sequence[ShadowJob]) -> list[dict]:
    """Score and log each job (one INFO line each). Never raises."""
    records = []
    for job in jobs:
        record = score_job(job)
        records.append(record)
        try:
            logger.info("[RerankShadow] %s", json.dumps(record, sort_keys=True))
        except Exception:
            pass
    return records


def score_jobs_async(jobs: Sequence[ShadowJob]) -> threading.Thread | None:
    """Score ``jobs`` on a daemon thread; ``None`` when there is nothing to score."""
    if not jobs:
        return None
    try:
        thread = threading.Thread(target=score_jobs, args=(list(jobs),), name="RerankShadow", daemon=True)
        thread.start()
        return thread
    except Exception:
        logger.debug("[RerankShadow] could not start the scoring thread", exc_info=True)
        return None
