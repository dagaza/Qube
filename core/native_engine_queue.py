"""
Priority command queue for NativeLlamaEngine.

Lower priority number = higher precedence. FIFO within the same priority via seq.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional


class EnginePriority(IntEnum):
    interactive = 0
    background = 1
    maintenance = 2
    control = 3


_PRIORITY_NAMES: dict[int, str] = {
    EnginePriority.interactive: "interactive",
    EnginePriority.background: "background",
    EnginePriority.maintenance: "maintenance",
    EnginePriority.control: "control",
}


def priority_for_op(op: str, *, task: Any = None) -> EnginePriority:
    if op in ("load", "unload", "shutdown"):
        return EnginePriority.control
    if op == "profile_behavior":
        return EnginePriority.maintenance
    if op == "generate":
        return EnginePriority.interactive
    if op == "chat_once":
        return EnginePriority.background
    return EnginePriority.background


def priority_label(priority: int) -> str:
    return _PRIORITY_NAMES.get(int(priority), f"priority_{priority}")


@dataclass(order=True)
class _HeapEntry:
    priority: int
    seq: int
    cmd: dict


class PriorityCommandQueue:
    """Thread-safe min-heap queue ordered by (priority, seq)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heap: list[_HeapEntry] = []
        self._seq_counter = itertools.count()
        self._not_empty = threading.Condition(self._lock)

    def put(self, cmd: dict, *, priority: EnginePriority | int) -> dict:
        """Enqueue cmd; stamp priority/seq/submitted_at. Returns the stored cmd."""
        with self._not_empty:
            seq = next(self._seq_counter)
            stamped = dict(cmd)
            stamped["priority"] = int(priority)
            stamped["seq"] = seq
            stamped.setdefault("submitted_at", time.monotonic())
            heapq.heappush(self._heap, _HeapEntry(int(priority), seq, stamped))
            self._not_empty.notify()
            return stamped

    def get(self, timeout: float | None = None) -> dict:
        with self._not_empty:
            if not self._heap:
                if timeout is None:
                    while not self._heap:
                        self._not_empty.wait()
                else:
                    deadline = time.monotonic() + timeout
                    while not self._heap:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise queue.Empty
                        self._not_empty.wait(remaining)
            entry = heapq.heappop(self._heap)
            cmd = dict(entry.cmd)
            cmd["dequeued_at"] = time.monotonic()
            return cmd

    def purge(self, predicate: Callable[[dict], bool]) -> int:
        """Remove entries matching predicate. Returns count removed."""
        with self._not_empty:
            before = len(self._heap)
            self._heap = [e for e in self._heap if not predicate(e.cmd)]
            heapq.heapify(self._heap)
            removed = before - len(self._heap)
            if removed:
                self._not_empty.notify_all()
            return removed

    def qsize(self) -> int:
        with self._lock:
            return len(self._heap)

    def count(self, predicate: Callable[[dict], bool]) -> int:
        with self._lock:
            return sum(1 for entry in self._heap if predicate(entry.cmd))

    def depth_by_priority(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for entry in self._heap:
                label = priority_label(entry.priority)
                counts[label] = counts.get(label, 0) + 1
            return counts

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            callers: list[str] = []
            for entry in sorted(self._heap):
                cmd = entry.cmd
                label = priority_label(entry.priority)
                counts[label] = counts.get(label, 0) + 1
                caller = str(cmd.get("debug_caller") or cmd.get("op") or "unknown")
                callers.append(caller)
            return {
                "depth_total": len(self._heap),
                "depth_by_priority": counts,
                "queued_callers": callers,
            }


# queue.Empty for get(timeout=...) parity with stdlib queue.Queue
import queue  # noqa: E402

__all__ = [
    "EnginePriority",
    "PriorityCommandQueue",
    "priority_for_op",
    "priority_label",
]
