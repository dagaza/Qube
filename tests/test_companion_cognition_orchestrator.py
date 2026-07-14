"""End-to-end tests for Companion Cognition orchestrator."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from core.assistant_activity import AssistantActivity, user_presence_label
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_cognition.orchestrator import CompanionCognitionOrchestrator
from core.companion_verbal_policy import CompanionVerbalGateContext
from core.platform.companion_capabilities import CompanionPlatformTier


def _ctx() -> CompanionVerbalGateContext:
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
    return CompanionVerbalGateContext(
        snapshot=snap,
        companion_visible=True,
        idle_since=time.time() - 120,
        main_window_visible=False,
        now=time.time(),
    )


class TestCompanionCognitionOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        from core.companion_cognition import message_library as ml
        import core.companion_cognition.variety as variety_mod

        ml._library_cache = ml.load_message_library(ml.bundled_messages_path())
        variety_mod._global_store = variety_mod.VarietyStore()

    @patch("core.companion_cognition.orchestrator.app_settings.get_companion_enabled", return_value=True)
    @patch("core.companion_cognition.orchestrator.app_settings.get_companion_verbal_enabled", return_value=True)
    def test_test_trigger_returns_local_l0(self, *_mocks) -> None:
        orch = CompanionCognitionOrchestrator()
        result = orch.process_legacy("test", _ctx(), sidecar_available=False)
        self.assertIsNone(result.skip_reason or None if result.local else result.skip_reason)
        self.assertIsNotNone(result.local)
        assert result.local is not None
        self.assertGreater(len(result.local.line), 3)
        self.assertEqual(result.local.level.value, 0)

    @patch("core.companion_cognition.orchestrator.app_settings.get_companion_enabled", return_value=True)
    @patch("core.companion_cognition.orchestrator.app_settings.get_companion_verbal_enabled", return_value=True)
    def test_ingest_produces_local_line(self, *_mocks) -> None:
        orch = CompanionCognitionOrchestrator()
        result = orch.process_legacy(
            "ingest_complete",
            _ctx(),
            sidecar_available=False,
            file_count=2,
        )
        self.assertIsNotNone(result.local, result.skip_reason)
        assert result.local is not None
        self.assertIn(result.local.intent, ("acknowledge_effort",))


if __name__ == "__main__":
    unittest.main()
