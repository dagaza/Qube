"""
Thread-safe queue of memory usage events.

Phase C of the memory enrichment hardening. Two producers and one consumer
share this queue:

- Producer 1: ``mcp.memory_tool.memory_search`` enqueues ``("retrieved",
  memory_lance_id)`` for every memory entry that is included in the
  retrieval context for a given turn.
- Producer 2: ``workers.llm_worker._execute_llm_turn`` scans the final
  generated text for ``[N]`` citation tokens, maps each to the lance row
  id of the memory source it represents, and enqueues
  ``("cited", memory_lance_id)`` for those.
- Consumer: ``workers.enrichment_worker.EnrichmentWorker`` drains the
  queue on its event loop and applies the increments to the LanceDB
  ``text`` payload via the existing safe delete+re-add path.

Producers MUST NOT do disk I/O on the retrieval / generation hot path —
this queue exists exactly so the vector search and the streaming loop
never block on row updates. The consumer batches drains so we don't write
amplify per-event.

Everything here is intentionally allocation-light: a single
``queue.Queue`` and short tuples. No external locks needed.
"""
from __future__ import annotations

import logging
from queue import Empty, Queue
from typing import Iterable, Optional

logger = logging.getLogger("Qube.MemoryUsageRecorder")


# Sentinel kinds for the queue.
KIND_RETRIEVED = "retrieved"
KIND_CITED = "cited"


class MemoryUsageRecorder:
    """Process-wide singleton-style recorder for memory usage events."""

    __slots__ = ("_q", "_max")

    def __init__(self, maxsize: int = 1024) -> None:
        self._q: Queue = Queue(maxsize=maxsize)
        self._max = int(maxsize)

    def record_retrieved(self, memory_id: str) -> None:
        if not memory_id:
            return
        try:
            self._q.put_nowait((KIND_RETRIEVED, str(memory_id)))
        except Exception:
            # Queue full: silently drop. We will simply miss a counter bump,
            # which is acceptable; no caller depends on a perfect count.
            pass

    def record_cited(self, memory_id: str) -> None:
        if not memory_id:
            return
        try:
            self._q.put_nowait((KIND_CITED, str(memory_id)))
        except Exception:
            pass

    def drain(self, max_items: int = 256) -> list[tuple[str, str]]:
        """Pop up to ``max_items`` events. Returns ``[]`` when empty."""
        out: list[tuple[str, str]] = []
        for _ in range(max(1, int(max_items))):
            try:
                item = self._q.get_nowait()
            except Empty:
                break
            out.append(item)
        return out

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except Exception:
            return 0


# ============================================================
# Process-wide singleton accessor.
# ============================================================
_RECORDER: Optional[MemoryUsageRecorder] = None


def get_memory_usage_recorder() -> MemoryUsageRecorder:
    """Return the process-wide :class:`MemoryUsageRecorder` instance."""
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = MemoryUsageRecorder()
    return _RECORDER


__all__ = [
    "MemoryUsageRecorder",
    "get_memory_usage_recorder",
    "KIND_RETRIEVED",
    "KIND_CITED",
]
