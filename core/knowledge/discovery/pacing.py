"""Global pacing between live DuckDuckGo discovery HTTP requests."""

from __future__ import annotations

import os
import random
import threading
import time

DEFAULT_DISCOVERY_PACE_MIN_SEC = 3.0
DEFAULT_DISCOVERY_PACE_JITTER_MIN_SEC = 0.5
DEFAULT_DISCOVERY_PACE_JITTER_MAX_SEC = 1.5
DEFAULT_DISCOVERY_PACE_MAX_WAIT_SEC = 30.0

_lock = threading.Lock()
_last_ddg_request_at: float = 0.0


def discovery_pacing_enabled() -> bool:
    raw = os.getenv("QUBE_DISCOVERY_PACE_ENABLED")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    try:
        from core.app_settings import get_discovery_pacing_enabled

        return bool(get_discovery_pacing_enabled())
    except Exception:
        return True


def effective_discovery_pace_min_seconds() -> float:
    """Base pacing interval multiplied by conservative health mode when active."""
    base = discovery_pace_min_seconds()
    if base <= 0:
        return 0.0
    try:
        from core.knowledge.discovery.health import get_conservative_pacing_multiplier

        return base * get_conservative_pacing_multiplier()
    except Exception:
        return base


def discovery_pace_min_seconds() -> float:
    raw = os.getenv("QUBE_DISCOVERY_PACE_MIN_SEC")
    if raw is None:
        return DEFAULT_DISCOVERY_PACE_MIN_SEC
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return DEFAULT_DISCOVERY_PACE_MIN_SEC


def discovery_pace_jitter_seconds() -> tuple[float, float]:
    raw_min = os.getenv("QUBE_DISCOVERY_PACE_JITTER_MIN_SEC")
    raw_max = os.getenv("QUBE_DISCOVERY_PACE_JITTER_MAX_SEC")
    try:
        jitter_min = (
            float(str(raw_min).strip())
            if raw_min is not None
            else DEFAULT_DISCOVERY_PACE_JITTER_MIN_SEC
        )
    except ValueError:
        jitter_min = DEFAULT_DISCOVERY_PACE_JITTER_MIN_SEC
    try:
        jitter_max = (
            float(str(raw_max).strip())
            if raw_max is not None
            else DEFAULT_DISCOVERY_PACE_JITTER_MAX_SEC
        )
    except ValueError:
        jitter_max = DEFAULT_DISCOVERY_PACE_JITTER_MAX_SEC
    if jitter_max < jitter_min:
        jitter_min, jitter_max = jitter_max, jitter_min
    return max(0.0, jitter_min), max(0.0, jitter_max)


def discovery_pace_max_wait_seconds() -> float:
    raw = os.getenv("QUBE_DISCOVERY_PACE_MAX_WAIT_SEC")
    if raw is None:
        return DEFAULT_DISCOVERY_PACE_MAX_WAIT_SEC
    try:
        return max(1.0, float(str(raw).strip()))
    except ValueError:
        return DEFAULT_DISCOVERY_PACE_MAX_WAIT_SEC


def wait_for_ddg_pace_slot(
    *,
    max_wait_sec: float | None = None,
) -> tuple[bool, int]:
    """Block until a DDG pace slot is available.

    Returns (acquired, wait_ms). When acquired is False, the caller should
    skip the live DDG HTTP call (pacing timeout).
    """
    if not discovery_pacing_enabled():
        return True, 0

    global _last_ddg_request_at

    min_sec = effective_discovery_pace_min_seconds()
    if min_sec <= 0:
        return True, 0

    jitter_min, jitter_max = discovery_pace_jitter_seconds()
    required_gap = min_sec + (
        random.uniform(jitter_min, jitter_max) if jitter_max > 0 else 0.0
    )
    deadline = time.time() + (max_wait_sec or discovery_pace_max_wait_seconds())
    started = time.time()

    while True:
        with _lock:
            now = time.time()
            elapsed = now - _last_ddg_request_at
            if _last_ddg_request_at <= 0 or elapsed >= required_gap:
                _last_ddg_request_at = now
                return True, int((now - started) * 1000)

        if time.time() >= deadline:
            return False, int((time.time() - started) * 1000)

        time.sleep(0.2)


def reset_discovery_pacing() -> None:
    """Clear pacing state (tests only)."""
    global _last_ddg_request_at
    with _lock:
        _last_ddg_request_at = 0.0
