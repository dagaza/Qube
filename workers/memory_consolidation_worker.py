"""
Memory v7.1 consolidation worker — deterministic cross-day staging (no LLM).
"""
from __future__ import annotations

import json
import logging
import random
import time

from PyQt6.QtCore import QMutex, QMutexLocker, QThread

from core.app_settings import get_enable_memory_consolidation
from core.memory_consolidation import should_stage_for_consolidation

logger = logging.getLogger("Qube.MemoryConsolidationWorker")

CONSOLIDATE_INTERVAL_SEC = 6 * 60 * 60.0
BATCH_SIZE = 15
LOOP_TICK_SEC = 30.0
SCAN_LIMIT = 500

# In-memory counters for optional telemetry UI.
_last_run_ts: float = 0.0
_last_staged_count: int = 0


def get_consolidation_telemetry() -> dict:
    return {
        "last_run_ts": _last_run_ts,
        "last_staged_count": _last_staged_count,
    }


class MemoryConsolidationWorker(QThread):
    """Stage context/knowledge rows that show durable cross-day retrieval patterns."""

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._running = True
        self._enabled_mutex = QMutex()
        self._is_enabled = get_enable_memory_consolidation()
        self._next_run_at = time.time() + random.uniform(180.0, 540.0)

    def set_enabled(self, enabled: bool) -> None:
        with QMutexLocker(self._enabled_mutex):
            self._is_enabled = bool(enabled)

    def _is_enabled_read(self) -> bool:
        with QMutexLocker(self._enabled_mutex):
            return self._is_enabled

    def shutdown(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            try:
                now = time.time()
                if now >= self._next_run_at and self._is_enabled_read():
                    self._run_cycle()
                    self._next_run_at = time.time() + CONSOLIDATE_INTERVAL_SEC
            except Exception as e:
                logger.exception("[MemoryConsolidation] cycle failed: %s", e)
            for _ in range(int(LOOP_TICK_SEC)):
                if not self._running:
                    return
                time.sleep(1.0)

    def _run_cycle(self) -> None:
        global _last_run_ts, _last_staged_count
        time.sleep(random.uniform(0.5, 2.5))
        try:
            rows = (
                self.store.table.search()
                .where("source LIKE 'qube_memory::%'")
                .limit(SCAN_LIMIT)
                .to_list()
            )
        except Exception as e:
            logger.debug("[MemoryConsolidation] scan failed: %s", e)
            return

        now = time.time()
        staged = 0
        for row in rows[:BATCH_SIZE * 4]:
            if staged >= BATCH_SIZE:
                break
            source = str(row.get("source") or "")
            try:
                payload = json.loads(row.get("text", "{}") or "{}")
            except Exception:
                continue
            ok, score, hints = should_stage_for_consolidation(payload, source, now=now)
            if not ok:
                continue
            payload["consolidation_score"] = round(score, 4)
            payload["consolidation_hints"] = hints
            payload["consolidation_staged_at"] = int(now)
            if self._rewrite_row(row, payload):
                staged += 1

        _last_run_ts = now
        _last_staged_count = staged
        if staged:
            logger.info("[Memory v7.1] consolidation staged %d row(s)", staged)

    def _rewrite_row(self, row: dict, payload: dict) -> bool:
        row_id = row.get("id")
        if not row_id:
            return False
        try:
            safe_id = str(row_id).replace("'", "''")
            self.store.table.delete(f"id = '{safe_id}'")
            self.store.table.add(
                [
                    {
                        "text": json.dumps(payload),
                        "vector": row.get("vector"),
                        "source": row.get("source"),
                        "chunk_id": int(row.get("chunk_id") or 0),
                    }
                ]
            )
            return True
        except Exception as e:
            logger.debug("[MemoryConsolidation] rewrite failed: %s", e)
            return False
