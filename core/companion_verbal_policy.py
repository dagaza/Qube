"""Policy and rate limits for companion verbal commentary."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from core import app_settings
from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_policy import (
    CompanionSuppressReason,
    companion_attention_mode,
    plan_companion_visibility,
)


class CompanionVerbalFrequency(str, Enum):
    RARE = "rare"
    NORMAL = "normal"
    CHATTY = "chatty"


DEFAULT_COMPANION_VERBAL_FREQUENCY = CompanionVerbalFrequency.NORMAL

_FREQUENCY_IDLE_MIN_SEC: dict[CompanionVerbalFrequency, int] = {
    CompanionVerbalFrequency.RARE: 45 * 60,
    CompanionVerbalFrequency.NORMAL: 15 * 60,
    CompanionVerbalFrequency.CHATTY: 5 * 60,
}

_FREQUENCY_IDLE_BEFORE_SEC: dict[CompanionVerbalFrequency, int] = {
    CompanionVerbalFrequency.RARE: 120,
    CompanionVerbalFrequency.NORMAL: 60,
    CompanionVerbalFrequency.CHATTY: 30,
}


class CompanionVerbalTrigger(str, Enum):
    IDLE = "idle"
    INGEST_COMPLETE = "ingest_complete"
    DOWNLOAD_COMPLETE = "download_complete"
    MODEL_LOADED = "model_loaded"
    STARTUP = "startup"
    MILESTONE = "milestone"
    USAGE_PATTERN = "long_term_usage_patterns"


def normalize_companion_verbal_frequency(
    value: str | CompanionVerbalFrequency | None,
) -> CompanionVerbalFrequency:
    if isinstance(value, CompanionVerbalFrequency):
        return value
    raw = str(value or "").strip().lower()
    for freq in CompanionVerbalFrequency:
        if freq.value == raw:
            return freq
    return DEFAULT_COMPANION_VERBAL_FREQUENCY


def frequency_idle_min_interval_sec(frequency: CompanionVerbalFrequency | str) -> int:
    return _FREQUENCY_IDLE_MIN_SEC[normalize_companion_verbal_frequency(frequency)]


def frequency_idle_before_sec(frequency: CompanionVerbalFrequency | str) -> int:
    return _FREQUENCY_IDLE_BEFORE_SEC[normalize_companion_verbal_frequency(frequency)]


def frequency_idle_label(frequency: CompanionVerbalFrequency | str) -> str:
    """User-facing summary of idle spacing for settings tooltips."""
    freq = normalize_companion_verbal_frequency(frequency)
    before = frequency_idle_before_sec(freq)
    interval = frequency_idle_min_interval_sec(freq)
    before_label = f"{before // 60} min" if before >= 60 else f"{before} sec"
    interval_label = f"{interval // 60} min" if interval >= 60 else f"{interval} sec"
    return (
        f"After {before_label} of assistant idle, at most one line every {interval_label}"
    )


def frequency_event_min_interval_sec(frequency: CompanionVerbalFrequency | str) -> int:
    return max(60, frequency_idle_min_interval_sec(frequency) // 2)


@dataclass
class CompanionVerbalRateLimiter:
    """In-memory emit timestamps (resets on app restart)."""

    last_idle_emit: float = 0.0
    last_event_emit: float = 0.0

    def can_emit_idle(self, *, now: float, min_interval_sec: float) -> bool:
        if self.last_idle_emit <= 0:
            return True
        return (now - self.last_idle_emit) >= min_interval_sec

    def can_emit_event(self, *, now: float, min_interval_sec: float) -> bool:
        if self.last_event_emit <= 0 and self.last_idle_emit <= 0:
            return True
        last = max(self.last_idle_emit, self.last_event_emit)
        if last <= 0:
            return True
        return (now - last) >= min_interval_sec

    def record_idle(self, *, now: float) -> None:
        self.last_idle_emit = now

    def record_event(self, *, now: float) -> None:
        self.last_event_emit = now


@dataclass(frozen=True)
class CompanionVerbalGateContext:
    snapshot: AssistantPresenceSnapshot
    companion_visible: bool
    idle_since: float | None
    snooze_until: float = 0.0
    main_window_visible: bool = False
    main_window_minimized: bool = False
    companion_user_visible: bool = True
    fullscreen_detected: bool = False
    now: float = field(default_factory=time.time)


def _verbal_master_enabled() -> bool:
    return (
        app_settings.get_companion_enabled()
        and app_settings.get_companion_verbal_enabled()
    )


def _base_gates(ctx: CompanionVerbalGateContext) -> tuple[bool, str]:
    if not _verbal_master_enabled():
        return False, "disabled"
    if companion_attention_mode(ctx.snapshot):
        return False, "attention_mode"
    if ctx.snapshot.activity != AssistantActivity.IDLE_LISTEN:
        return False, "not_idle"
    if app_settings.get_notifications_dnd() or ctx.snapshot.dnd:
        return False, "dnd"

    plan = plan_companion_visibility(
        ctx.snapshot,
        main_window_visible=ctx.main_window_visible,
        main_window_minimized=ctx.main_window_minimized,
        companion_user_visible=ctx.companion_user_visible,
        fullscreen_detected=ctx.fullscreen_detected,
        snooze_until=ctx.snooze_until,
        now=ctx.now,
    )
    if not plan.show:
        reason = plan.reason.value if plan.reason != CompanionSuppressReason.NONE else "hidden"
        return False, reason
    if not ctx.companion_visible:
        return False, "not_visible"
    return True, ""


def should_emit_idle(
    ctx: CompanionVerbalGateContext,
    limiter: CompanionVerbalRateLimiter,
) -> bool:
    ok, _ = _base_gates(ctx)
    if not ok:
        return False
    # Idle quips while the main window is foreground only when the user keeps the companion visible there.
    if ctx.main_window_visible and not ctx.main_window_minimized:
        if not app_settings.get_companion_show_while_window_open():
            return False
    if ctx.idle_since is None:
        return False
    freq = normalize_companion_verbal_frequency(app_settings.get_companion_verbal_frequency())
    if (ctx.now - ctx.idle_since) < frequency_idle_before_sec(freq):
        return False
    return limiter.can_emit_idle(
        now=ctx.now,
        min_interval_sec=frequency_idle_min_interval_sec(freq),
    )


def _normalize_trigger(value: CompanionVerbalTrigger | str) -> CompanionVerbalTrigger | None:
    if isinstance(value, CompanionVerbalTrigger):
        return value
    raw = str(value or "").strip().lower()
    for trig in CompanionVerbalTrigger:
        if trig.value == raw:
            return trig
    return None


def _base_gates_event(ctx: CompanionVerbalGateContext) -> tuple[bool, str]:
    """Event commentary may fire while the user is not in IDLE_LISTEN (e.g. after ingest)."""
    if not _verbal_master_enabled():
        return False, "disabled"
    if companion_attention_mode(ctx.snapshot):
        return False, "attention_mode"
    if app_settings.get_notifications_dnd() or ctx.snapshot.dnd:
        return False, "dnd"

    plan = plan_companion_visibility(
        ctx.snapshot,
        main_window_visible=ctx.main_window_visible,
        main_window_minimized=ctx.main_window_minimized,
        companion_user_visible=ctx.companion_user_visible,
        fullscreen_detected=ctx.fullscreen_detected,
        snooze_until=ctx.snooze_until,
        now=ctx.now,
    )
    if not plan.show:
        reason = plan.reason.value if plan.reason != CompanionSuppressReason.NONE else "hidden"
        return False, reason
    if not ctx.companion_visible:
        return False, "not_visible"
    return True, ""


def should_emit_event(
    trigger: CompanionVerbalTrigger | str,
    ctx: CompanionVerbalGateContext,
    limiter: CompanionVerbalRateLimiter,
) -> bool:
    trig = _normalize_trigger(trigger)
    if trig is None:
        return False
    if trig == CompanionVerbalTrigger.INGEST_COMPLETE:
        if not app_settings.get_companion_verbal_react_ingest():
            return False
    elif trig == CompanionVerbalTrigger.DOWNLOAD_COMPLETE:
        if not app_settings.get_companion_verbal_react_download():
            return False
    elif trig in (
        CompanionVerbalTrigger.MODEL_LOADED,
        CompanionVerbalTrigger.STARTUP,
        CompanionVerbalTrigger.MILESTONE,
        CompanionVerbalTrigger.USAGE_PATTERN,
    ):
        pass
    else:
        return False

    ok, _ = _base_gates_event(ctx)
    if not ok:
        return False

    freq = normalize_companion_verbal_frequency(app_settings.get_companion_verbal_frequency())
    return limiter.can_emit_event(
        now=ctx.now,
        min_interval_sec=frequency_event_min_interval_sec(freq),
    )


def record_emitted(
    trigger: CompanionVerbalTrigger | str,
    limiter: CompanionVerbalRateLimiter,
    *,
    now: float | None = None,
) -> None:
    ts = now if now is not None else time.time()
    trig = _normalize_trigger(trigger)
    if trig is None:
        return
    if trig == CompanionVerbalTrigger.IDLE:
        limiter.record_idle(now=ts)
    else:
        limiter.record_event(now=ts)


def record_cognition_emission(
    trigger: CompanionVerbalTrigger | str,
    limiter: CompanionVerbalRateLimiter,
    *,
    message_id: str = "",
    now: float | None = None,
) -> None:
    """Extended emit record — message_id reserved for variety store (orchestrator)."""
    record_emitted(trigger, limiter, now=now)


def should_show_companion_line(
    trigger: str,
    snapshot: AssistantPresenceSnapshot,
) -> bool:
    """Re-check before displaying a completed sidecar line."""
    trig = str(trigger or "idle").strip().lower()
    if companion_attention_mode(snapshot):
        return False
    if trig == "idle" and snapshot.activity != AssistantActivity.IDLE_LISTEN:
        return False
    return True
