"""
Persistent negative-pattern list for the memory subsystem.

Stores embeddings of memories the user explicitly deleted (or flagged) in
the Memory Manager. The extraction pipeline rejects new candidates whose
content vector is closer than ``DEFAULT_REJECT_DISTANCE`` to any entry in
the list, preventing the same offending memory from being re-created the
next time the user has a similar conversation.

Persisted as a small JSON file at ``~/.qube/memory_negatives.json`` so the
list survives app restarts and (deliberately) is independent of the
LanceDB store, mirroring the pattern used by ``model_overrides.json``.

The vector column is stored as a plain Python list of floats — embeddings
are small enough that JSON is fine and the format is human-inspectable.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger("Qube.MemoryNegativeList")


def _default_path() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".qube", "memory_negatives.json")


# Distance below which an extraction candidate is rejected.
# Same metric as LanceDB (L2 over normalized Nomic v1.5 vectors), so a
# value of 0.20 corresponds to "very similar" but not "near-duplicate".
DEFAULT_REJECT_DISTANCE = 0.20

# Hard cap on the number of negatives we keep — old entries get rotated
# out FIFO once we hit the cap. Keeps the per-extraction reject scan O(n)
# at a manageable n.
MAX_NEGATIVES = 500


class MemoryNegativeList:
    """In-memory cache + JSON persistence for the negative pattern list."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or _default_path()
        self._lock = threading.RLock()
        self._entries: list[dict] = []
        self._vectors: Optional[np.ndarray] = None
        self._load()

    # --------------------------- public API ---------------------------

    def add(self, content: str, vector) -> None:
        """Add a deleted memory to the negative list and persist."""
        if not content:
            return
        try:
            v = np.asarray(vector, dtype=np.float32)
        except Exception:
            return
        if v.ndim != 1 or v.size == 0:
            return

        entry = {
            "content": content,
            "vector": v.tolist(),
            "ts": int(time.time()),
        }
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > MAX_NEGATIVES:
                self._entries = self._entries[-MAX_NEGATIVES:]
            self._rebuild_matrix()
            self._save()

    def is_negative(
        self,
        vector,
        threshold: float = DEFAULT_REJECT_DISTANCE,
    ) -> bool:
        """True when ``vector`` is closer than ``threshold`` to any
        recorded negative."""
        with self._lock:
            if self._vectors is None or self._vectors.shape[0] == 0:
                return False
            try:
                v = np.asarray(vector, dtype=np.float32)
            except Exception:
                return False
            if v.ndim != 1 or v.size != self._vectors.shape[1]:
                return False

            # L2 distance: same metric used elsewhere in the memory pipeline.
            diffs = self._vectors - v[None, :]
            dists = np.linalg.norm(diffs, axis=1)
            return bool(dists.min() < threshold)

    def all(self) -> list[dict]:
        with self._lock:
            return [
                {"content": e["content"], "ts": e.get("ts", 0)}
                for e in self._entries
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # --------------------------- internals ----------------------------

    def _rebuild_matrix(self) -> None:
        if not self._entries:
            self._vectors = None
            return
        try:
            self._vectors = np.asarray(
                [e["vector"] for e in self._entries], dtype=np.float32
            )
        except Exception:
            self._vectors = None

    def _load(self) -> None:
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            entries = data.get("entries") or []
            cleaned: list[dict] = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                v = e.get("vector")
                if not isinstance(v, list) or not v:
                    continue
                cleaned.append({
                    "content": str(e.get("content") or "")[:1000],
                    "vector": v,
                    "ts": int(e.get("ts") or 0),
                })
            with self._lock:
                self._entries = cleaned[-MAX_NEGATIVES:]
                self._rebuild_matrix()
            logger.info(
                "[MemoryNegativeList] loaded %d entries from %s",
                len(self._entries),
                self.path,
            )
        except Exception as e:
            logger.warning("[MemoryNegativeList] load failed: %s", e)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"entries": self._entries}, f)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.warning("[MemoryNegativeList] save failed: %s", e)


# ============================================================
# Process-wide singleton accessor.
# ============================================================
_INSTANCE: Optional[MemoryNegativeList] = None
_INSTANCE_LOCK = threading.Lock()


def get_memory_negative_list() -> MemoryNegativeList:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = MemoryNegativeList()
    return _INSTANCE


__all__ = [
    "MemoryNegativeList",
    "get_memory_negative_list",
    "DEFAULT_REJECT_DISTANCE",
]
