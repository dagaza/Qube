"""Companion verbal policy and rate limits."""
from __future__ import annotations

import time
import unittest
from unittest import mock

from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_verbal_policy import (
    CompanionVerbalFrequency,
    CompanionVerbalGateContext,
    CompanionVerbalRateLimiter,
    CompanionVerbalTrigger,
    frequency_event_min_interval_sec,
    frequency_idle_min_interval_sec,
    record_emitted,
    should_emit_event,
    should_emit_idle,
    should_show_companion_line,
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


class TestCompanionVerbalPolicy(unittest.TestCase):
    def test_frequency_intervals(self) -> None:
        self.assertEqual(frequency_idle_min_interval_sec(CompanionVerbalFrequency.RARE), 45 * 60)
        self.assertEqual(frequency_idle_min_interval_sec(CompanionVerbalFrequency.NORMAL), 15 * 60)
        self.assertEqual(frequency_event_min_interval_sec(CompanionVerbalFrequency.NORMAL), 450)

    def test_idle_requires_idle_since(self) -> None:
        limiter = CompanionVerbalRateLimiter()
        now = 1_000_000.0
        ctx = CompanionVerbalGateContext(
            snapshot=_snap(),
            companion_visible=True,
            idle_since=None,
            now=now,
        )
        with mock.patch("core.companion_verbal_policy.app_settings") as mock_settings:
            mock_settings.get_companion_enabled.return_value = True
            mock_settings.get_companion_verbal_enabled.return_value = True
            mock_settings.get_companion_verbal_frequency.return_value = "normal"
            mock_settings.get_notifications_dnd.return_value = False
            self.assertFalse(should_emit_idle(ctx, limiter))

    def test_idle_respects_rate_limit(self) -> None:
        limiter = CompanionVerbalRateLimiter()
        now = 2_000_000.0
        limiter.record_idle(now=now - 60)
        ctx = CompanionVerbalGateContext(
            snapshot=_snap(),
            companion_visible=True,
            idle_since=now - 120,
            now=now,
        )
        with mock.patch("core.companion_verbal_policy.app_settings") as mock_settings:
            mock_settings.get_companion_enabled.return_value = True
            mock_settings.get_companion_verbal_enabled.return_value = True
            mock_settings.get_companion_verbal_frequency.return_value = "normal"
            mock_settings.get_notifications_dnd.return_value = False
            self.assertFalse(should_emit_idle(ctx, limiter))

    def test_attention_mode_blocks_display(self) -> None:
        snap = _snap(AssistantActivity.WORKING)
        self.assertFalse(should_show_companion_line("idle", snap))

    def test_idle_blocked_while_main_window_open(self) -> None:
        limiter = CompanionVerbalRateLimiter()
        now = 4_000_000.0
        ctx = CompanionVerbalGateContext(
            snapshot=_snap(),
            companion_visible=True,
            idle_since=now - 120,
            main_window_visible=True,
            main_window_minimized=False,
            now=now,
        )
        with mock.patch("core.companion_verbal_policy.app_settings") as mock_settings:
            mock_settings.get_companion_enabled.return_value = True
            mock_settings.get_companion_verbal_enabled.return_value = True
            mock_settings.get_companion_verbal_frequency.return_value = "normal"
            mock_settings.get_notifications_dnd.return_value = False
            mock_settings.get_companion_show_while_window_open.return_value = False
            self.assertFalse(should_emit_idle(ctx, limiter))

    def test_idle_allowed_while_main_window_open_when_companion_shown(self) -> None:
        limiter = CompanionVerbalRateLimiter()
        now = 4_000_000.0
        ctx = CompanionVerbalGateContext(
            snapshot=_snap(),
            companion_visible=True,
            idle_since=now - 120,
            main_window_visible=True,
            main_window_minimized=False,
            now=now,
        )
        with mock.patch("core.companion_verbal_policy.app_settings") as mock_settings:
            mock_settings.get_companion_enabled.return_value = True
            mock_settings.get_companion_verbal_enabled.return_value = True
            mock_settings.get_companion_verbal_frequency.return_value = "normal"
            mock_settings.get_notifications_dnd.return_value = False
            mock_settings.get_companion_show_while_window_open.return_value = True
            self.assertTrue(should_emit_idle(ctx, limiter))

    def test_record_emitted_updates_limiter(self) -> None:
        limiter = CompanionVerbalRateLimiter()
        ts = 3_000_000.0
        record_emitted(CompanionVerbalTrigger.IDLE, limiter, now=ts)
        self.assertEqual(limiter.last_idle_emit, ts)
        record_emitted(CompanionVerbalTrigger.INGEST_COMPLETE, limiter, now=ts + 1)
        self.assertEqual(limiter.last_event_emit, ts + 1)


if __name__ == "__main__":
    unittest.main()
