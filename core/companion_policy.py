"""Pure policy rules for desktop companion visibility and auto-hide."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from core import app_settings
from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot
from core.platform.companion_capabilities import CompanionPlatformTier


class CompanionSuppressReason(str, Enum):
    NONE = ""
    DISABLED = "disabled"
    PLATFORM_NONE = "platform_none"
    PLATFORM_DEGRADED_OFF = "platform_degraded_off"
    MAIN_WINDOW_VISIBLE = "main_window_visible"
    SNOOZED = "snoozed"
    FULLSCREEN = "fullscreen"
    DND = "dnd"
    USER_HIDDEN = "user_hidden"


@dataclass(frozen=True)
class CompanionVisibilityPlan:
    show: bool
    auto_hide: bool
    passive_click_through: bool
    use_dock_mode: bool
    reason: CompanionSuppressReason = CompanionSuppressReason.NONE


def _attention_activity(activity: AssistantActivity) -> bool:
    return activity in (
        AssistantActivity.CAPTURING,
        AssistantActivity.WORKING,
        AssistantActivity.SPEAKING,
        AssistantActivity.NEEDS_ATTENTION,
        AssistantActivity.ERROR,
    )


def should_auto_hide(snapshot: AssistantPresenceSnapshot, *, idle_since: float | None) -> bool:
    if not app_settings.get_companion_auto_hide_idle():
        return False
    if _attention_activity(snapshot.activity):
        return False
    if snapshot.activity == AssistantActivity.ASSISTANT_OFF:
        return True
    if snapshot.activity == AssistantActivity.BACKGROUND_BUSY:
        return True
    if snapshot.activity != AssistantActivity.IDLE_LISTEN:
        return False
    if idle_since is None:
        return False
    return (time.time() - idle_since) >= app_settings.get_companion_idle_fade_sec()


def plan_companion_visibility(
    snapshot: AssistantPresenceSnapshot,
    *,
    main_window_visible: bool,
    main_window_minimized: bool,
    companion_user_visible: bool,
    fullscreen_detected: bool,
    snooze_until: float,
    now: float | None = None,
) -> CompanionVisibilityPlan:
    """Decide whether the companion window should be shown."""
    ts = now if now is not None else time.time()

    if not app_settings.get_companion_enabled():
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.DISABLED)

    tier = snapshot.platform_tier
    if tier == CompanionPlatformTier.NONE:
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.PLATFORM_NONE)

    if tier == CompanionPlatformTier.DEGRADED and not app_settings.get_companion_try_on_wayland():
        return CompanionVisibilityPlan(False, False, False, True, CompanionSuppressReason.PLATFORM_DEGRADED_OFF)

    if snooze_until > ts:
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.SNOOZED)

    if not companion_user_visible:
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.USER_HIDDEN)

    show_while_open = app_settings.get_companion_show_while_window_open()
    main_open = main_window_visible and not main_window_minimized

    if main_open and not show_while_open:
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.MAIN_WINDOW_VISIBLE)

    if not main_open and not app_settings.get_companion_show_when_tray_hidden():
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.MAIN_WINDOW_VISIBLE)

    if fullscreen_detected and app_settings.get_companion_suppress_on_fullscreen():
        if not _attention_activity(snapshot.activity):
            return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.FULLSCREEN)

    if snapshot.dnd and snapshot.activity not in (
        AssistantActivity.NEEDS_ATTENTION,
        AssistantActivity.ERROR,
    ):
        return CompanionVisibilityPlan(False, False, False, False, CompanionSuppressReason.DND)

    use_dock = tier == CompanionPlatformTier.DEGRADED and app_settings.get_companion_dock_mode()
    auto_hide = should_auto_hide(snapshot, idle_since=None)
    passive = snapshot.activity == AssistantActivity.IDLE_LISTEN and auto_hide

    return CompanionVisibilityPlan(True, auto_hide, passive, use_dock)


def companion_attention_mode(snapshot: AssistantPresenceSnapshot) -> bool:
    """True when companion is actively conveying assistant state (suppress duplicate OS notify)."""
    return _attention_activity(snapshot.activity) or snapshot.attention_required
