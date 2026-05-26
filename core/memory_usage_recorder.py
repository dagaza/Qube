"""
Thread-safe queue of memory usage events.

Phase C of the memory enrichment hardening. Two producers and one consumer
share this queue:

- Producer 1: ``mcp.memory_tool.memory_search`` enqueues retrieved events
  for every memory entry included in the retrieval context.
- Producer 2: ``workers.llm_worker._execute_llm_turn`` enqueues citations.
- Consumer: ``workers.enrichment_worker.EnrichmentWorker`` drains the queue.

Producers MUST NOT do disk I/O on the retrieval / generation hot path.
"""
from __future__ import annotations

import hashlib
import logging
from queue import Empty, Queue
from typing import Optional

logger = logging.getLogger("Qube.MemoryUsageRecorder")

KIND_RETRIEVED = "retrieved"
KIND_CITED = "cited"

# kind, memory_id, query_fingerprint, retrieval_score
RetrievalEvent = tuple[str, str, Optional[str], Optional[float]]


def compute_query_fingerprint(query: str, **tier_flags: bool) -> str:
    """Stable short hash for promotion query-diversity tracking."""
    parts = [str(query or "").strip().lower()]
    for key in sorted(tier_flags.keys()):
        parts.append(f"{key}={int(bool(tier_flags[key]))}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


class MemoryUsageRecorder:
    """Process-wide singleton-style recorder for memory usage events."""

    __slots__ = ("_q", "_max")

    def __init__(self, maxsize: int = 1024) -> None:
        self._q: Queue = Queue(maxsize=maxsize)
        self._max = int(maxsize)

    def record_retrieved(
        self,
        memory_id: str,
        *,
        query_fingerprint: str | None = None,
        retrieval_score: float | None = None,
    ) -> None:
        if not memory_id:
            return
        score: Optional[float] = None
        if retrieval_score is not None:
            try:
                score = max(0.0, float(retrieval_score))
            except (TypeError, ValueError):
                score = None
        try:
            self._q.put_nowait((KIND_RETRIEVED, str(memory_id), query_fingerprint, score))
        except Exception:
            pass

    def record_cited(self, memory_id: str) -> None:
        if not memory_id:
            return
        try:
            self._q.put_nowait((KIND_CITED, str(memory_id), None, None))
        except Exception:
            pass

    def drain(self, max_items: int = 256) -> list[RetrievalEvent]:
        """Pop up to ``max_items`` events. Returns ``[]`` when empty."""
        out: list[RetrievalEvent] = []
        for _ in range(max(1, int(max_items))):
            try:
                item = self._q.get_nowait()
            except Empty:
                break
            if len(item) == 2:
                kind, mid = item
                out.append((kind, mid, None, None))
            elif len(item) == 3:
                kind, mid, fp = item
                out.append((kind, mid, fp, None))
            else:
                kind, mid, fp, score = item
                out.append((kind, mid, fp, score))
        return out

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except Exception:
            return 0


_RECORDER: Optional[MemoryUsageRecorder] = None


def get_memory_usage_recorder() -> MemoryUsageRecorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = MemoryUsageRecorder()
    return _RECORDER


__all__ = [
    "MemoryUsageRecorder",
    "get_memory_usage_recorder",
    "compute_query_fingerprint",
    "KIND_RETRIEVED",
    "KIND_CITED",
    "RetrievalEvent",
]
