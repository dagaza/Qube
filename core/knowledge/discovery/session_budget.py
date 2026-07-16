"""Per-process rolling budgets for live DuckDuckGo SERP HTTP calls."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Literal

DEFAULT_DDG_BURST_BUDGET = 6
DEFAULT_DDG_BURST_WINDOW_SEC = 600
DEFAULT_DDG_SESSION_BUDGET = 30
DEFAULT_DDG_SESSION_BUDGET_WINDOW_SEC = 3600

BudgetBlockReason = Literal["burst", "session"]

_lock = threading.Lock()
_live_request_times: list[float] = []


def ddg_session_budget_enabled() -> bool:
    raw = os.getenv("QUBE_DDG_SESSION_BUDGET_ENABLED")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def ddg_burst_budget_enabled() -> bool:
    raw = os.getenv("QUBE_DDG_BURST_BUDGET_ENABLED")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def ddg_burst_budget_limit() -> int:
    raw = os.getenv("QUBE_DDG_BURST_BUDGET")
    if raw is None:
        return DEFAULT_DDG_BURST_BUDGET
    try:
        value = int(str(raw).strip())
        if value <= 0:
            return 0
        return value
    except ValueError:
        return DEFAULT_DDG_BURST_BUDGET


def ddg_burst_budget_window_seconds() -> int:
    raw = os.getenv("QUBE_DDG_BURST_WINDOW_SEC")
    if raw is None:
        return DEFAULT_DDG_BURST_WINDOW_SEC
    try:
        return max(60, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_DDG_BURST_WINDOW_SEC


def ddg_session_budget_limit() -> int:
    raw = os.getenv("QUBE_DDG_SESSION_BUDGET")
    if raw is None:
        try:
            from core.app_settings import get_ddg_session_budget_override

            override = get_ddg_session_budget_override()
            if override > 0:
                return override
        except Exception:
            pass
        return DEFAULT_DDG_SESSION_BUDGET
    try:
        value = int(str(raw).strip())
        if value <= 0:
            return 0
        return value
    except ValueError:
        return DEFAULT_DDG_SESSION_BUDGET


def ddg_session_budget_window_seconds() -> int:
    raw = os.getenv("QUBE_DDG_SESSION_BUDGET_WINDOW_SEC")
    if raw is None:
        return DEFAULT_DDG_SESSION_BUDGET_WINDOW_SEC
    try:
        return max(60, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_DDG_SESSION_BUDGET_WINDOW_SEC


@dataclass(frozen=True)
class DdgBurstBudgetStatus:
    used: int
    limit: int
    remaining: int
    window_seconds: int

    @property
    def exhausted(self) -> bool:
        if self.limit <= 0:
            return False
        return self.used >= self.limit


@dataclass(frozen=True)
class DdgSessionBudgetStatus:
    used: int
    limit: int
    remaining: int
    window_seconds: int

    @property
    def exhausted(self) -> bool:
        if self.limit <= 0:
            return False
        return self.used >= self.limit


def _prune_to_session_window_locked(now: float) -> None:
    cutoff = now - ddg_session_budget_window_seconds()
    while _live_request_times and _live_request_times[0] < cutoff:
        _live_request_times.pop(0)


def get_ddg_burst_budget_status() -> DdgBurstBudgetStatus:
    limit = ddg_burst_budget_limit()
    window = ddg_burst_budget_window_seconds()
    with _lock:
        now = time.time()
        _prune_to_session_window_locked(now)
        burst_cutoff = now - window
        used = sum(1 for ts in _live_request_times if ts >= burst_cutoff)
    remaining = max(0, limit - used) if limit > 0 else 0
    return DdgBurstBudgetStatus(
        used=used,
        limit=limit,
        remaining=remaining,
        window_seconds=window,
    )


def get_ddg_session_budget_status() -> DdgSessionBudgetStatus:
    limit = ddg_session_budget_limit()
    window = ddg_session_budget_window_seconds()
    with _lock:
        now = time.time()
        _prune_to_session_window_locked(now)
        used = len(_live_request_times)
    remaining = max(0, limit - used) if limit > 0 else 0
    return DdgSessionBudgetStatus(
        used=used,
        limit=limit,
        remaining=remaining,
        window_seconds=window,
    )


def is_ddg_burst_budget_exhausted() -> bool:
    if not ddg_burst_budget_enabled():
        return False
    status = get_ddg_burst_budget_status()
    if status.limit <= 0:
        return False
    return status.exhausted


def is_ddg_session_budget_exhausted() -> bool:
    if not ddg_session_budget_enabled():
        return False
    status = get_ddg_session_budget_status()
    if status.limit <= 0:
        return False
    return status.exhausted


def get_ddg_budget_block_reason() -> BudgetBlockReason | None:
    if is_ddg_burst_budget_exhausted():
        return "burst"
    if is_ddg_session_budget_exhausted():
        return "session"
    return None


def record_ddg_live_request() -> None:
    """Record a live DDG HTTP request (not cache hits or skipped calls)."""
    with _lock:
        now = time.time()
        _prune_to_session_window_locked(now)
        _live_request_times.append(now)


def _burst_window_minutes() -> int:
    return max(1, (ddg_burst_budget_window_seconds() + 59) // 60)


def _session_window_minutes() -> int:
    return max(1, (ddg_session_budget_window_seconds() + 59) // 60)


def format_burst_budget_summary() -> str | None:
    if not ddg_burst_budget_enabled():
        return None
    status = get_ddg_burst_budget_status()
    if status.limit <= 0:
        return None
    minutes = _burst_window_minutes()
    if status.exhausted:
        return (
            f"DuckDuckGo burst limit reached "
            f"({status.used}/{status.limit} live queries in {minutes} min); using fallbacks."
        )
    return (
        f"DuckDuckGo burst window ({minutes} min): {status.used}/{status.limit} live queries"
    )


def format_session_budget_summary() -> str | None:
    if not ddg_session_budget_enabled():
        return None
    status = get_ddg_session_budget_status()
    if status.limit <= 0:
        return None
    minutes = _session_window_minutes()
    if status.exhausted:
        return (
            f"DuckDuckGo session limit reached "
            f"({status.used}/{status.limit} live queries in {minutes} min); using fallbacks."
        )
    return (
        f"DuckDuckGo session window ({minutes} min): {status.used}/{status.limit} live queries"
    )


def format_discovery_budget_usage_lines() -> list[str]:
    """Short read-only usage lines for Settings."""
    lines: list[str] = []
    burst = format_burst_budget_summary()
    session = format_session_budget_summary()
    if burst:
        lines.append(burst)
    if session:
        lines.append(session)
    return lines


def discovery_budget_log_fields() -> dict[str, int]:
    burst = get_ddg_burst_budget_status()
    session = get_ddg_session_budget_status()
    return {
        "burst_used": burst.used,
        "burst_limit": burst.limit,
        "burst_remaining": burst.remaining,
        "session_used": session.used,
        "session_limit": session.limit,
        "session_remaining": session.remaining,
    }


def reset_ddg_session_budget() -> None:
    """Clear budget counters (tests only)."""
    with _lock:
        _live_request_times.clear()
