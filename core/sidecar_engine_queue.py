"""
Priority command queue for SidecarLlmWorker.

Lower priority number = higher precedence. FIFO within the same priority via seq.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

from core.native_engine_queue import PriorityCommandQueue, priority_label
from core.sidecar_types import SidecarTask

# Phase C — background burst caps
COMPANION_DEFER_QUEUE_DEPTH = 8
INGEST_BLURB_MAX_QUEUED = 12

_FOREGROUND_TASKS = frozenset(
    {SidecarTask.query_rewrite, SidecarTask.source_digest}
)


class SidecarPriority(IntEnum):
    interactive = 0
    background = 1
    control = 2


_PRIORITY_NAMES: dict[int, str] = {
    SidecarPriority.interactive: "interactive",
    SidecarPriority.background: "background",
    SidecarPriority.control: "control",
}


def sidecar_priority_label(priority: int) -> str:
    return _PRIORITY_NAMES.get(int(priority), priority_label(priority))


def should_defer_companion_line(queue_depth: int) -> bool:
    return int(queue_depth) >= COMPANION_DEFER_QUEUE_DEPTH


def should_drop_ingest_blurb(pending_ingest_count: int) -> bool:
    return int(pending_ingest_count) >= INGEST_BLURB_MAX_QUEUED


def priority_for_sidecar_cmd(cmd: dict) -> SidecarPriority:
    op = str(cmd.get("op") or "")
    if op in ("shutdown", "reload"):
        return SidecarPriority.control
    if op == "task":
        task = cmd.get("task")
        if isinstance(task, SidecarTask) and task in _FOREGROUND_TASKS:
            return SidecarPriority.interactive
        if isinstance(task, str) and task in {t.value for t in _FOREGROUND_TASKS}:
            return SidecarPriority.interactive
    return SidecarPriority.background


class SidecarCommandQueue:
    """Thin wrapper stamping sidecar-specific priorities on ``PriorityCommandQueue``."""

    def __init__(self) -> None:
        self._inner = PriorityCommandQueue()

    def put(self, cmd: dict) -> dict:
        priority = priority_for_sidecar_cmd(cmd)
        return self._inner.put(cmd, priority=priority)

    def get(self, timeout: float | None = None) -> dict:
        return self._inner.get(timeout=timeout)

    def purge(self, predicate) -> int:
        return self._inner.purge(predicate)

    def count(self, predicate) -> int:
        return self._inner.count(predicate)

    def qsize(self) -> int:
        return self._inner.qsize()

    def snapshot(self) -> dict[str, Any]:
        snap = self._inner.snapshot()
        by_pri = snap.get("depth_by_priority") or {}
        remapped: dict[str, int] = {}
        for key, val in by_pri.items():
            if key == "interactive":
                remapped["interactive"] = int(val)
            elif key == "background":
                remapped["background"] = int(val)
            elif key == "control":
                remapped["control"] = int(val)
            else:
                remapped[key] = int(val)
        snap["depth_by_priority"] = remapped
        return snap

    def depth_by_priority(self) -> dict[str, int]:
        raw = self._inner.depth_by_priority()
        out: dict[str, int] = {}
        for pri, count in raw.items():
            if pri == "interactive":
                out["interactive"] = count
            elif pri == "background":
                out["background"] = count
            elif pri == "control":
                out["control"] = count
            else:
                out[pri] = count
        return out


__all__ = [
    "COMPANION_DEFER_QUEUE_DEPTH",
    "INGEST_BLURB_MAX_QUEUED",
    "SidecarCommandQueue",
    "SidecarPriority",
    "priority_for_sidecar_cmd",
    "should_defer_companion_line",
    "should_drop_ingest_blurb",
    "sidecar_priority_label",
]
