"""Tests for companion visibility policy."""

from __future__ import annotations

import time

from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_policy import (
    CompanionSuppressReason,
    companion_attention_mode,
    plan_companion_visibility,
    should_auto_hide,
)
from core.platform.companion_capabilities import CompanionPlatformTier


def _snap(activity: AssistantActivity = AssistantActivity.IDLE_LISTEN) -> AssistantPresenceSnapshot:
    return AssistantPresenceSnapshot(
        activity=activity,
        phase=None,
        display_text="",
        bubble_state="idle",
        voice_input_paused=False,
        voice_output_muted=False,
        dnd=False,
        background_busy=False,
        caption_text=None,
        attention_required=False,
        platform_tier=CompanionPlatformTier.FULL,
    )


def test_companion_hidden_when_disabled(monkeypatch):
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_enabled", lambda: False)
    plan = plan_companion_visibility(
        _snap(),
        main_window_visible=False,
        main_window_minimized=False,
        companion_user_visible=True,
        fullscreen_detected=False,
        snooze_until=0.0,
    )
    assert plan.show is False
    assert plan.reason == CompanionSuppressReason.DISABLED


def test_companion_shows_when_tray_hidden(monkeypatch):
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_enabled", lambda: True)
    monkeypatch.setattr(
        "core.companion_policy.app_settings.get_companion_show_when_tray_hidden", lambda: True
    )
    monkeypatch.setattr(
        "core.companion_policy.app_settings.get_companion_show_while_window_open", lambda: False
    )
    monkeypatch.setattr(
        "core.companion_policy.app_settings.get_companion_try_on_wayland", lambda: True
    )
    plan = plan_companion_visibility(
        _snap(),
        main_window_visible=False,
        main_window_minimized=False,
        companion_user_visible=True,
        fullscreen_detected=False,
        snooze_until=0.0,
    )
    assert plan.show is True


def test_companion_hidden_when_main_open(monkeypatch):
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_enabled", lambda: True)
    monkeypatch.setattr(
        "core.companion_policy.app_settings.get_companion_show_while_window_open", lambda: False
    )
    plan = plan_companion_visibility(
        _snap(),
        main_window_visible=True,
        main_window_minimized=False,
        companion_user_visible=True,
        fullscreen_detected=False,
        snooze_until=0.0,
    )
    assert plan.show is False
    assert plan.reason == CompanionSuppressReason.MAIN_WINDOW_VISIBLE


def test_wayland_degraded_requires_opt_in(monkeypatch):
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_enabled", lambda: True)
    monkeypatch.setattr(
        "core.companion_policy.app_settings.get_companion_try_on_wayland", lambda: False
    )
    snap = _snap()
    snap = AssistantPresenceSnapshot(
        **{**snap.__dict__, "platform_tier": CompanionPlatformTier.DEGRADED}
    )
    plan = plan_companion_visibility(
        snap,
        main_window_visible=False,
        main_window_minimized=False,
        companion_user_visible=True,
        fullscreen_detected=False,
        snooze_until=0.0,
    )
    assert plan.show is False
    assert plan.use_dock_mode is True


def test_auto_hide_idle_after_threshold(monkeypatch):
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_auto_hide_idle", lambda: True)
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_idle_fade_sec", lambda: 5)
    idle_since = time.time() - 10
    assert should_auto_hide(_snap(), idle_since=idle_since) is True


def test_no_auto_hide_while_capturing(monkeypatch):
    monkeypatch.setattr("core.companion_policy.app_settings.get_companion_auto_hide_idle", lambda: True)
    snap = _snap(AssistantActivity.CAPTURING)
    assert should_auto_hide(snap, idle_since=time.time() - 100) is False


def test_companion_attention_mode_active_during_working():
    snap = _snap(AssistantActivity.WORKING)
    assert companion_attention_mode(snap) is True


def test_companion_attention_mode_idle_false():
    assert companion_attention_mode(_snap()) is False
