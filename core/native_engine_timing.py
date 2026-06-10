"""
Timing helpers for NativeLlamaEngine job observability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


def _ms(start: float | None, end: float | None) -> int:
    if start is None or end is None:
        return 0
    return max(0, int(round((end - start) * 1000)))


@dataclass
class EngineJobTiming:
    request_id: str = ""
    task_type: str = ""
    debug_caller: str = ""
    exchange_id: Optional[int] = None
    priority: str = ""
    queue_depth_at_submit: int = 0
    queue_depth_at_start: int = 0
    queued_behind: list[str] = field(default_factory=list)
    submitted_at: float = 0.0
    dequeued_at: Optional[float] = None
    inference_started_at: Optional[float] = None
    inference_finished_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancelled: bool = False
    preempted_by: Optional[str] = None
    session_id: str = ""
    reschedule_attempt: int = 0

    @property
    def queue_wait_ms(self) -> int:
        return _ms(self.submitted_at, self.dequeued_at)

    @property
    def engine_prep_ms(self) -> int:
        return _ms(self.dequeued_at, self.inference_started_at)

    @property
    def inference_ms(self) -> int:
        return _ms(self.inference_started_at, self.inference_finished_at)

    @property
    def total_ms(self) -> int:
        end = self.finished_at or self.inference_finished_at
        return _ms(self.submitted_at, end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_type": self.task_type,
            "debug_caller": self.debug_caller,
            "exchange_id": self.exchange_id,
            "priority": self.priority,
            "queue_depth_at_submit": self.queue_depth_at_submit,
            "queue_depth_at_start": self.queue_depth_at_start,
            "queued_behind": list(self.queued_behind),
            "queue_wait_ms": self.queue_wait_ms,
            "engine_prep_ms": self.engine_prep_ms,
            "inference_ms": self.inference_ms,
            "total_ms": self.total_ms,
            "cancelled": self.cancelled,
            "preempted_by": self.preempted_by,
            "session_id": self.session_id or None,
            "reschedule_attempt": self.reschedule_attempt or None,
        }


def timing_from_cmd(cmd: dict, *, finished_at: float | None = None) -> EngineJobTiming:
    task = cmd.get("task")
    task_type = (
        str(getattr(task, "value", task) or "")
        if task is not None
        else str(cmd.get("op") or "")
    )
    meta = cmd.get("_timing_meta") or {}
    return EngineJobTiming(
        request_id=str(cmd.get("request_id") or ""),
        task_type=task_type,
        debug_caller=str(cmd.get("debug_caller") or cmd.get("op") or ""),
        exchange_id=cmd.get("debug_exchange_id"),
        priority=str(cmd.get("priority_label") or cmd.get("priority") or ""),
        queue_depth_at_submit=int(meta.get("queue_depth_at_submit") or 0),
        queue_depth_at_start=int(meta.get("queue_depth_at_start") or 0),
        queued_behind=list(meta.get("queued_behind") or []),
        submitted_at=float(cmd.get("submitted_at") or 0.0),
        dequeued_at=cmd.get("dequeued_at"),
        inference_started_at=cmd.get("inference_started_at"),
        inference_finished_at=cmd.get("inference_finished_at"),
        finished_at=finished_at or time.monotonic(),
        cancelled=bool(cmd.get("cancelled")),
        preempted_by=cmd.get("preempted_by"),
        session_id=str(cmd.get("session_id") or ""),
        reschedule_attempt=int(cmd.get("reschedule_attempt") or 0),
    )
