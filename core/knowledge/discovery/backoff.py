"""Temporary backoff for discovery providers after bot challenges."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

DEFAULT_DDG_BOT_BACKOFF_SECONDS = 1800

_lock = threading.Lock()
_entries: dict[str, tuple[float, str]] = {}
_pending_ddg_backoff_notification = False


def discovery_backoff_enabled() -> bool:
    raw = os.getenv("QUBE_DISCOVERY_BACKOFF")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def ddg_bot_backoff_seconds() -> int:
    raw = os.getenv("QUBE_DDG_BACKOFF_SECONDS")
    if raw is None:
        return DEFAULT_DDG_BOT_BACKOFF_SECONDS
    try:
        return max(60, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_DDG_BOT_BACKOFF_SECONDS


@dataclass(frozen=True)
class DiscoveryBackoffEntry:
    provider_id: str
    reason: str
    expires_at: float

    @property
    def remaining_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


def mark_provider_backoff(
    provider_id: str,
    *,
    reason: str = "bot_challenge",
    ttl_seconds: int | None = None,
) -> bool:
    """Pause a provider; returns True when backoff was newly activated."""
    global _pending_ddg_backoff_notification
    if not discovery_backoff_enabled():
        return False
    pid = (provider_id or "").strip().lower()
    if not pid:
        return False
    ttl = ttl_seconds
    if ttl is None and pid == "duckduckgo":
        ttl = ddg_bot_backoff_seconds()
    ttl = max(0, int(ttl or 0))
    if ttl <= 0:
        return False
    expires_at = time.time() + ttl
    with _lock:
        existing = _entries.get(pid)
        already_active = (
            existing is not None and existing[0] > time.time()
        )
        _entries[pid] = (expires_at, reason)
        newly_activated = not already_active
        if newly_activated and pid == "duckduckgo":
            _pending_ddg_backoff_notification = True
        return newly_activated


def get_provider_backoff(provider_id: str) -> DiscoveryBackoffEntry | None:
    pid = (provider_id or "").strip().lower()
    if not pid or not discovery_backoff_enabled():
        return None
    with _lock:
        row = _entries.get(pid)
        if row is None:
            return None
        expires_at, reason = row
        if time.time() >= expires_at:
            _entries.pop(pid, None)
            return None
        return DiscoveryBackoffEntry(
            provider_id=pid,
            reason=reason,
            expires_at=expires_at,
        )


def is_provider_in_backoff(provider_id: str) -> bool:
    return get_provider_backoff(provider_id) is not None


def clear_provider_backoff(provider_id: str) -> None:
    pid = (provider_id or "").strip().lower()
    if not pid:
        return
    with _lock:
        _entries.pop(pid, None)


def reset_discovery_backoff() -> None:
    """Clear all backoff entries (tests only)."""
    global _pending_ddg_backoff_notification
    with _lock:
        _entries.clear()
        _pending_ddg_backoff_notification = False


def consume_ddg_backoff_notification() -> tuple[bool, int]:
    """Return (should_notify_user, remaining_seconds) once per new DDG backoff."""
    global _pending_ddg_backoff_notification
    with _lock:
        pending = _pending_ddg_backoff_notification
        _pending_ddg_backoff_notification = False
    if not pending:
        return False, 0
    entry = get_provider_backoff("duckduckgo")
    if entry is None:
        return False, 0
    return True, entry.remaining_seconds


def format_backoff_summary(entry: DiscoveryBackoffEntry | None) -> str | None:
    if entry is None:
        return None
    minutes = max(1, (entry.remaining_seconds + 59) // 60)
    label = entry.provider_id
    if entry.reason == "bot_challenge":
        return (
            f"{label} paused for ~{minutes} min after bot challenge "
            "(using fallbacks; no further DDG requests until pause ends)."
        )
    return f"{label} paused for ~{minutes} min ({entry.reason})."
