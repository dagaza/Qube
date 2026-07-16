"""Adaptive discovery health — conservative pacing after repeated DDG challenges (R9)."""

from __future__ import annotations

import threading
import time

CHALLENGE_WINDOW_SECONDS = 86400
CONSERVATIVE_PACING_THRESHOLD = 2
TIER_B_SUGGEST_THRESHOLD = 3
CONSERVATIVE_DURATION_SECONDS = 86400

_lock = threading.Lock()
_challenge_timestamps: list[float] = []
_conservative_until: float = 0.0
_pending_tier_b_suggestion = False


def _prune_locked(now: float) -> None:
    cutoff = now - CHALLENGE_WINDOW_SECONDS
    while _challenge_timestamps and _challenge_timestamps[0] < cutoff:
        _challenge_timestamps.pop(0)


def record_ddg_bot_challenge() -> None:
    """Track a DDG bot challenge for conservative pacing and tier suggestions."""
    global _pending_tier_b_suggestion, _conservative_until
    now = time.time()
    with _lock:
        _prune_locked(now)
        _challenge_timestamps.append(now)
        count = len(_challenge_timestamps)
        if count >= CONSERVATIVE_PACING_THRESHOLD:
            _conservative_until = now + CONSERVATIVE_DURATION_SECONDS
        if count >= TIER_B_SUGGEST_THRESHOLD:
            from core.knowledge.discovery.privacy_policy import (
                TIER_PRIVATE,
                get_active_privacy_tier,
            )

            if get_active_privacy_tier() == TIER_PRIVATE:
                _pending_tier_b_suggestion = True


def record_ddg_serp_success() -> None:
    """Clear challenge streak after a successful live DDG SERP."""
    global _conservative_until
    with _lock:
        _challenge_timestamps.clear()
        _conservative_until = 0.0


def challenge_count_24h() -> int:
    with _lock:
        now = time.time()
        _prune_locked(now)
        return len(_challenge_timestamps)


def get_conservative_pacing_multiplier() -> float:
    if time.time() < _conservative_until:
        return 2.0
    return 1.0


def is_conservative_mode_active() -> bool:
    return get_conservative_pacing_multiplier() > 1.0


def conservative_mode_summary() -> str | None:
    if not is_conservative_mode_active():
        return None
    remaining = max(0, int(_conservative_until - time.time()))
    hours = max(1, (remaining + 3599) // 3600)
    return (
        f"Conservative pacing active (~{hours}h remaining) after repeated DDG "
        f"challenges ({challenge_count_24h()} in 24h)."
    )


def consume_tier_b_suggestion() -> bool:
    """Return True once when Tier B should be suggested to the user."""
    global _pending_tier_b_suggestion
    with _lock:
        pending = _pending_tier_b_suggestion
        _pending_tier_b_suggestion = False
    return pending


def reset_discovery_health() -> None:
    """Clear challenge history and conservative pacing (Settings action)."""
    global _pending_tier_b_suggestion, _conservative_until
    with _lock:
        _challenge_timestamps.clear()
        _conservative_until = 0.0
        _pending_tier_b_suggestion = False


def reset_discovery_health_state() -> None:
    """Alias for tests."""
    reset_discovery_health()
