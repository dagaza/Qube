"""Short-TTL negative cache for throttled / budget-exhausted knowledge hosts."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Literal

NegativeReason = Literal["budget_exhausted", "circuit_open"]

DEFAULT_NEGATIVE_TTL_SECONDS = 300

_lock = threading.Lock()
_entries: dict[str, tuple[float, NegativeReason]] = {}


def negative_cache_enabled() -> bool:
    raw = os.getenv("QUBE_NEGATIVE_CACHE")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def negative_cache_ttl_seconds() -> int:
    raw = os.getenv("QUBE_NEGATIVE_CACHE_TTL")
    if raw is None:
        return DEFAULT_NEGATIVE_TTL_SECONDS
    try:
        return max(0, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_NEGATIVE_TTL_SECONDS


@dataclass(frozen=True)
class NegativeCacheEntry:
    host: str
    reason: NegativeReason
    expires_at: float


def mark_host_negative(
    host: str,
    *,
    reason: NegativeReason,
    ttl_seconds: int | None = None,
) -> None:
    """Remember a host as temporarily unavailable (in-process, thread-safe)."""
    if not negative_cache_enabled():
        return
    key = (host or "").strip().lower()
    if not key:
        return
    ttl = negative_cache_ttl_seconds() if ttl_seconds is None else max(0, ttl_seconds)
    if ttl <= 0:
        return
    expires_at = time.time() + ttl
    with _lock:
        _entries[key] = (expires_at, reason)


def get_host_negative(host: str) -> NegativeCacheEntry | None:
    key = (host or "").strip().lower()
    if not key or not negative_cache_enabled():
        return None
    with _lock:
        row = _entries.get(key)
        if row is None:
            return None
        expires_at, reason = row
        if time.time() >= expires_at:
            _entries.pop(key, None)
            return None
        return NegativeCacheEntry(host=key, reason=reason, expires_at=expires_at)


def clear_host_negative(host: str) -> None:
    key = (host or "").strip().lower()
    if not key:
        return
    with _lock:
        _entries.pop(key, None)


def reset_negative_cache() -> None:
    """Clear all entries (tests only)."""
    with _lock:
        _entries.clear()
