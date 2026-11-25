"""Deadline-bounded scholarly adapter collection and multi-query merge (plan S8).

``collect_adapter_rows`` runs one search per (query variant, adapter) task in a
thread pool and stops waiting for:

* a task that has been running for ``adapter_timeout_s`` (``adapter_timeout``);
* every task still pending once ``deadline_s`` has passed since the start
  (``deadline``; a task that never got a worker is reported ``started=False``).

A late task's thread is not killed (Python cannot), it finishes in the
background and its rows are discarded; the HTTP client's own timeout bounds it.
Adapter exceptions are recorded instead of being swallowed (§3.7).

``rrf_merge`` fuses the per-task ranked lists with reciprocal-rank fusion and
de-duplicates rows by DOI or normalised title, so a paper found by two adapters
or two wordings is one candidate ranked by both appearances.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

VARIANT_PRIMARY = "primary"
VARIANT_RAW = "raw"
VARIANT_SPELLING = "spelling"

TIMEOUT_ADAPTER = "adapter_timeout"
TIMEOUT_DEADLINE = "deadline"

RRF_K = 60
DEFAULT_MAX_WORKERS = 6

SearchFn = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class AdapterTask:
    adapter_id: str
    variant: str
    query: str


@dataclass
class CollectionResult:
    adapter_calls: list[str] = field(default_factory=list)  # adapters that returned rows
    raw_audit: list[dict[str, Any]] = field(default_factory=list)
    ranked_lists: list[list[dict[str, Any]]] = field(default_factory=list)  # task order
    timeouts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    timings: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def timed_out_adapters(self) -> list[str]:
        return list(dict.fromkeys(t["adapter"] for t in self.timeouts))

    def diag(self) -> dict[str, Any]:
        return {
            "elapsed_ms": round(self.elapsed_ms, 1),
            "tasks": self.timings,
            "adapter_timeouts": self.timeouts,
            "adapter_errors": self.errors,
        }


def collect_adapter_rows(
    tasks: Sequence[AdapterTask],
    *,
    per_adapter: int,
    adapter_timeout_s: float | None,
    deadline_s: float | None,
    search_fn_for: Callable[[str], SearchFn | None],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> CollectionResult:
    """Run ``tasks`` in parallel under the per-task timeout and overall deadline."""
    out = CollectionResult()
    runnable: list[tuple[AdapterTask, SearchFn]] = []
    for task in tasks:
        fn = search_fn_for(task.adapter_id)
        if fn is not None and str(task.query or "").strip():
            runnable.append((task, fn))
    if not runnable:
        return out

    t0 = time.monotonic()
    started: dict[int, float] = {}
    lock = threading.Lock()
    results: dict[int, list[dict[str, Any]]] = {}

    def _run(index: int, task: AdapterTask, fn: SearchFn) -> list[dict[str, Any]]:
        with lock:
            started[index] = time.monotonic()
        rows = fn(task.query, max_results=per_adapter)
        return [dict(r) for r in (rows or []) if isinstance(r, dict)]

    pool = ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(runnable))))
    futures: dict[Future, int] = {}
    try:
        for index, (task, fn) in enumerate(runnable):
            futures[pool.submit(_run, index, task, fn)] = index
        pending = set(futures)
        while pending:
            now = time.monotonic()
            waits: list[float] = []
            if deadline_s is not None:
                waits.append(t0 + float(deadline_s) - now)
            if adapter_timeout_s is not None:
                with lock:
                    running = [started[futures[f]] for f in pending if futures[f] in started]
                if running:
                    waits.append(min(running) + float(adapter_timeout_s) - now)
            timeout = max(0.0, min(waits)) if waits else None
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            for fut in done:
                index = futures[fut]
                task = runnable[index][0]
                with lock:
                    began = started.get(index, t0)
                ms = (time.monotonic() - began) * 1000.0
                try:
                    rows = fut.result()
                except Exception as exc:  # recorded, not swallowed (§3.7)
                    rows = []
                    out.errors.append(
                        {
                            "adapter": task.adapter_id,
                            "variant": task.variant,
                            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                        }
                    )
                results[index] = rows
                out.timings.append(
                    {"adapter": task.adapter_id, "variant": task.variant, "ms": round(ms, 1), "rows": len(rows)}
                )
            now = time.monotonic()
            if adapter_timeout_s is not None:
                for fut in list(pending):
                    index = futures[fut]
                    with lock:
                        began = started.get(index)
                    if began is not None and now - began >= float(adapter_timeout_s):
                        pending.discard(fut)
                        fut.cancel()
                        out.timeouts.append(_timeout_entry(runnable[index][0], TIMEOUT_ADAPTER, now - began, True))
            if deadline_s is not None and now - t0 >= float(deadline_s) and pending:
                for fut in pending:
                    index = futures[fut]
                    with lock:
                        began = started.get(index)
                    fut.cancel()
                    out.timeouts.append(
                        _timeout_entry(
                            runnable[index][0],
                            TIMEOUT_DEADLINE,
                            (now - began) if began is not None else 0.0,
                            began is not None,
                        )
                    )
                pending = set()
    finally:
        # Never block the turn on a late adapter; its thread finishes on its own.
        pool.shutdown(wait=False, cancel_futures=True)

    for index, (task, _fn) in enumerate(runnable):
        rows = results.get(index)
        if not rows:
            continue
        for row in rows:
            row.setdefault("_query_variant", task.variant)
        out.adapter_calls.append(task.adapter_id)
        out.raw_audit.extend(dict(r) for r in rows)
        out.ranked_lists.append(rows)
    out.adapter_calls = list(dict.fromkeys(out.adapter_calls))
    out.elapsed_ms = (time.monotonic() - t0) * 1000.0
    return out


def _timeout_entry(task: AdapterTask, reason: str, elapsed_s: float, started: bool) -> dict[str, Any]:
    return {
        "adapter": task.adapter_id,
        "variant": task.variant,
        "reason": reason,
        "elapsed_ms": round(elapsed_s * 1000.0, 1),
        "started": started,
    }


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        keys.append(f"doi:{doi}")
    title = _normalize_title(str(row.get("title") or ""))
    if title:
        keys.append(f"title:{title}")
    return keys


def rrf_merge(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion of ``ranked_lists`` with DOI / title de-duplication.

    The first-seen copy of a paper is kept (lists are passed in task order, so
    the primary wording's copy wins); ``_query_variants`` lists every wording
    that found it and ``_rrf_score`` its fused score. Ties keep first-seen order.
    """
    canonical: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    variants: list[list[str]] = []
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked):
            keys = _row_keys(row)
            slot = next((canonical[key] for key in keys if key in canonical), None)
            if slot is None:
                slot = len(rows)
                rows.append(dict(row))
                scores.append(0.0)
                variants.append([])
            for key in keys:
                canonical.setdefault(key, slot)
            scores[slot] += 1.0 / (k + rank + 1)
            variant = str(row.get("_query_variant") or VARIANT_PRIMARY)
            if variant not in variants[slot]:
                variants[slot].append(variant)
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
    merged: list[dict[str, Any]] = []
    for i in order:
        row = rows[i]
        row["_rrf_score"] = round(scores[i], 6)
        row["_query_variants"] = list(variants[i])
        merged.append(row)
    return merged[:limit] if limit else merged


__all__ = [
    "AdapterTask",
    "CollectionResult",
    "TIMEOUT_ADAPTER",
    "TIMEOUT_DEADLINE",
    "VARIANT_PRIMARY",
    "VARIANT_RAW",
    "VARIANT_SPELLING",
    "collect_adapter_rows",
    "rrf_merge",
]
