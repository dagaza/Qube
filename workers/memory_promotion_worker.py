"""
Memory v7 promotion worker — deferred context/knowledge → preference tier.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Optional

from PyQt6.QtCore import QMutex, QMutexLocker, QThread

from core.lance_row_id import LANCE_ROW_ID_SELECT, lance_row_delete_filter, lance_row_id
from core.memory_filters import derive_memory_tier, is_memory_actionable, is_thin_content
from core.memory_negative_list import DEFAULT_REJECT_DISTANCE, get_memory_negative_list
from core.memory_promotion import (
    PROMOTION_NEAR_DUPLICATE_DISTANCE,
    compute_promotion_score,
    compute_promotion_signals,
    passes_promotion_gates_with_reason,
)
from core.app_settings import get_enable_memory_promotion, get_memory_promotion_preset

logger = logging.getLogger("Qube.MemoryPromotionWorker")

PROMOTE_INTERVAL_SEC = 6 * 60 * 60.0
BATCH_SIZE = 10
LOOP_TICK_SEC = 30.0
SCAN_LIMIT = 500


class MemoryPromotionWorker(QThread):
    """Periodic promotion of high-signal working-tier memories to preference."""

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._running = True
        self._enabled_mutex = QMutex()
        self._is_enabled = get_enable_memory_promotion()
        self._next_run_at = time.time() + random.uniform(120.0, 420.0)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with QMutexLocker(self._enabled_mutex):
            was = self._is_enabled
            self._is_enabled = enabled
            if was and not enabled:
                self._next_run_at = time.time() + PROMOTE_INTERVAL_SEC
        logger.debug("[MemoryPromotion] enabled=%s", enabled)

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
                    self._next_run_at = time.time() + PROMOTE_INTERVAL_SEC
            except Exception as e:
                logger.exception("[MemoryPromotion] cycle failed: %s", e)
            for _ in range(int(LOOP_TICK_SEC)):
                if not self._running:
                    return
                time.sleep(1.0)

    def _run_cycle(self) -> None:
        time.sleep(random.uniform(0.5, 2.0))
        candidates = self._scan_candidates()
        if not candidates:
            return
        promoted = 0
        for cand in candidates[:BATCH_SIZE]:
            if self._promote_one(cand):
                promoted += 1
        if promoted:
            logger.info("[Memory v7.1] promotion cycle promoted %d row(s)", promoted)

    def _scan_candidates(self) -> list[dict]:
        try:
            rows = (
                self.store.table.search()
                .select(LANCE_ROW_ID_SELECT)
                .where("source LIKE 'qube_memory::%'")
                .limit(SCAN_LIMIT)
                .to_list()
            )
        except Exception as e:
            logger.debug("[MemoryPromotion] scan failed: %s", e)
            return []

        preset = get_memory_promotion_preset()
        scored: list[tuple[float, dict]] = []
        now = time.time()
        for row in rows:
            source = str(row.get("source") or "")
            if "qube_memory::preference::" in source.lower():
                continue
            try:
                payload = json.loads(row.get("text", "{}") or "{}")
            except Exception:
                continue
            if payload.get("promoted_at"):
                continue
            ok, _reason, _ = passes_promotion_gates_with_reason(
                payload, source, now=now, preset=preset
            )
            if not ok:
                continue
            score = compute_promotion_score(payload, now=now)
            scored.append((score, {"id": lance_row_id(row), "source": source, "row": row, "payload": payload}))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored]

    def _fetch_live_row(self, row_id: str) -> Optional[dict]:
        if not row_id:
            return None
        row_filter = lance_row_delete_filter(row_id)
        if not row_filter:
            return None
        try:
            rows = (
                self.store.table.search()
                .select(LANCE_ROW_ID_SELECT)
                .where(row_filter)
                .limit(1)
                .to_list()
            )
            return rows[0] if rows else None
        except Exception:
            return None

    def _has_near_duplicate_preference(self, vector) -> bool:
        if vector is None:
            return False
        try:
            hits = (
                self.store.table.search(vector)
                .where("source LIKE 'qube_memory::preference::%'")
                .limit(1)
                .to_list()
            )
            if not hits:
                return False
            dist = hits[0].get("_distance", 1.0)
            return float(dist) < PROMOTION_NEAR_DUPLICATE_DISTANCE
        except Exception:
            return False

    def _promote_one(self, cand: dict) -> bool:
        row_id = str(cand.get("id") or "")
        live = self._fetch_live_row(row_id)
        if not live:
            return False

        row = live
        source = str(row.get("source") or cand.get("source") or "")
        try:
            payload = json.loads(row.get("text", "{}") or "{}")
        except Exception:
            return False

        now = time.time()
        preset = get_memory_promotion_preset()

        if payload.get("promoted_at"):
            return False
        if payload.get("flagged_for_review"):
            logger.debug("[MemoryPromotion] skip %s: reflection veto (flagged)", row_id[:8])
            return False

        if not is_memory_actionable(payload, now=now):
            return False
        content = (payload.get("content") or "").strip()
        if is_thin_content(content):
            return False

        try:
            vector = row.get("vector")
            neg = get_memory_negative_list()
            if neg.is_negative(vector, threshold=DEFAULT_REJECT_DISTANCE):
                return False
            if self._has_near_duplicate_preference(vector):
                logger.debug("[MemoryPromotion] skip %s: near-duplicate preference", row_id[:8])
                return False
        except Exception:
            pass

        ok, reason, _components = passes_promotion_gates_with_reason(
            payload, source, now=now, preset=preset
        )
        if not ok:
            logger.debug("[MemoryPromotion] skip %s: %s", row_id[:8], reason)
            return False

        category = str(payload.get("category") or "context").strip().lower() or "context"
        tier = derive_memory_tier(payload)
        if tier == "episode":
            return False

        new_source = f"qube_memory::preference::{category}"
        score = compute_promotion_score(payload, now=now)
        payload["promoted_at"] = int(now)
        payload["promotion_score"] = round(score, 4)
        payload["promotion_signals"] = compute_promotion_signals(payload, now=now)

        if not payload.get("first_seen_at"):
            payload["first_seen_at"] = payload.get("timestamp") or int(now)

        return self._rewrite_row(row, payload, new_source)

    def _rewrite_row(self, row: dict, payload: dict, new_source: str) -> bool:
        row_id = lance_row_id(row)
        delete_filter = lance_row_delete_filter(row_id)
        if not delete_filter:
            return False
        try:
            self.store.table.delete(delete_filter)
            self.store.table.add(
                [
                    {
                        "text": json.dumps(payload),
                        "vector": row.get("vector"),
                        "source": new_source,
                        "chunk_id": int(row.get("chunk_id") or 0),
                    }
                ]
            )
            logger.info(
                "[Memory v7.1] promoted memory %s -> %s (score=%.3f)",
                str(row_id)[:8],
                new_source,
                float(payload.get("promotion_score") or 0),
            )
            return True
        except Exception as e:
            logger.warning("[MemoryPromotion] rewrite failed: %s", e)
            return False
