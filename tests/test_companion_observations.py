"""Tests for companion observation engine."""

from __future__ import annotations

import time
import unittest

from core.assistant_activity import AssistantActivity, user_presence_label
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_cognition.observations import observe, trigger_event_from_legacy
from core.companion_cognition.types import CompanionTriggerEvent
from core.companion_verbal_policy import CompanionVerbalGateContext
from core.platform.companion_capabilities import CompanionPlatformTier


def _ctx(**kwargs) -> CompanionVerbalGateContext:
    snap = AssistantPresenceSnapshot(
        activity=AssistantActivity.IDLE_LISTEN,
        phase=None,
        display_text="",
        presence_label=user_presence_label(AssistantActivity.IDLE_LISTEN),
        bubble_state="idle",
        voice_input_paused=False,
        voice_output_muted=False,
        dnd=False,
        background_busy=False,
        caption_text=None,
        attention_required=False,
        platform_tier=CompanionPlatformTier.FULL,
    )
    defaults = dict(
        snapshot=snap,
        companion_visible=True,
        idle_since=time.time() - 120,
        now=time.time(),
    )
    defaults.update(kwargs)
    return CompanionVerbalGateContext(**defaults)


class TestCompanionObservations(unittest.TestCase):
    def test_idle_maps_to_quiet_period(self) -> None:
        event = trigger_event_from_legacy("idle")
        obs = observe(event, _ctx())
        self.assertIsNotNone(obs)
        assert obs is not None
        self.assertEqual(obs.type, "quiet_period")
        self.assertIn("idle_sec", obs.facts)

    def test_ingest_maps_with_file_count(self) -> None:
        event = trigger_event_from_legacy("ingest_complete", file_count=3)
        obs = observe(event, _ctx())
        assert obs is not None
        self.assertEqual(obs.type, "library_update_completed")
        self.assertEqual(obs.facts["file_count"], 3)

    def test_download_sanitizes_basename(self) -> None:
        event = trigger_event_from_legacy(
            "download_complete",
            basename="/evil/path/model-Q4_K_M.gguf",
        )
        obs = observe(event, _ctx())
        assert obs is not None
        self.assertEqual(obs.facts["basename"], "model-Q4_K_M.gguf")

    def test_rejects_disallowed_payload_key(self) -> None:
        event = CompanionTriggerEvent(
            source="idle",
            ts=time.time(),
            payload={"chat_history": "secret"},
        )
        self.assertIsNone(observe(event, _ctx()))

    def test_settings_preview(self) -> None:
        event = trigger_event_from_legacy("test")
        obs = observe(event, _ctx())
        assert obs is not None
        self.assertEqual(obs.type, "settings_preview")


if __name__ == "__main__":
    unittest.main()
