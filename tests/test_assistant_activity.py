"""Tests for assistant activity reducer."""

from unittest.mock import patch

from core.assistant_activity import (
    AssistantActivity,
    AssistantActivityReducer,
    menu_status_line,
    tray_tooltip_for_activity,
    user_presence_label,
)
from core.assistant_presence import (
    AssistantPresenceService,
    AssistantPhase,
    companion_status_caption,
    phase_from_message,
)


def test_reducer_blocks_stray_idle_during_listening():
    reducer = AssistantActivityReducer()
    reducer.reduce("Listening")
    blocked = reducer.reduce("Idle")
    assert blocked.blocked is True
    assert blocked.bubble_state == "listening"


def test_reducer_maps_legacy_recording_message_to_listening():
    reducer = AssistantActivityReducer()
    transition = reducer.reduce("🎙️ RECORDING...")
    assert transition.bubble_state == "listening"
    assert transition.display_text.strip() == "Listening"


def test_reducer_allows_voice_capture_idle():
    reducer = AssistantActivityReducer()
    reducer.reduce("Listening")
    ok = reducer.reduce("Voice capture idle")
    assert ok.blocked is False
    assert ok.bubble_state == "idle"
    assert ok.display_text.strip() == "Idle"


def test_stale_listening_for_phrase_after_capture_idle_stays_idle():
    """Regression: testbed-style copy must not re-enter capture after idle."""
    reducer = AssistantActivityReducer()
    reducer.reduce("Listening")
    reducer.reduce("Voice capture idle")
    late = reducer.reduce("Listening for jarvis...")
    assert late.bubble_state == "idle"
    assert late.activity == AssistantActivity.IDLE_LISTEN


def test_reducer_maps_thinking_to_working_activity():
    reducer = AssistantActivityReducer()
    t = reducer.reduce("Thinking...")
    assert t.activity == AssistantActivity.WORKING
    assert t.bubble_state == "thinking"
    assert t.display_text.strip() == "Thinking"


def test_reducer_maps_transcribing_to_thinking_display():
    reducer = AssistantActivityReducer()
    t = reducer.reduce("Transcribing...")
    assert t.activity == AssistantActivity.WORKING
    assert t.display_text.strip() == "Thinking"


def test_reducer_writing_when_voice_output_muted():
    reducer = AssistantActivityReducer()
    reducer.set_voice_output_muted(True)
    t = reducer.reduce("Thinking...")
    assert t.bubble_state == "writing"
    assert t.display_text.strip() == "Writing"


def test_reducer_blocks_stray_idle_during_thinking():
    reducer = AssistantActivityReducer()
    reducer.reduce("Thinking...")
    blocked = reducer.reduce("Idle")
    assert blocked.blocked is True
    assert blocked.bubble_state == "thinking"


def test_reducer_allows_forced_idle_after_thinking():
    reducer = AssistantActivityReducer()
    reducer.reduce("Thinking...")
    ok = reducer.reduce("Idle", force=True)
    assert ok.blocked is False
    assert ok.bubble_state == "idle"



def test_reducer_maps_voice_input_deactivated_to_idle():
    reducer = AssistantActivityReducer()
    t = reducer.reduce("Voice Input Deactivated", force=True)
    assert t.activity == AssistantActivity.IDLE_LISTEN
    assert t.bubble_state == "idle"
    assert t.display_text.strip() == "Idle"


def test_reducer_maps_native_model_ready_with_thinking_in_filename_to_idle():
    reducer = AssistantActivityReducer()
    t = reducer.reduce("Native model ready: Qwen3-32B-Thinking-Q4_K_M.gguf")
    assert t.activity == AssistantActivity.IDLE_LISTEN
    assert t.bubble_state == "idle"
    assert t.display_text.strip() == "Idle"


def test_reducer_maps_loading_native_model_to_idle():
    reducer = AssistantActivityReducer()
    t = reducer.reduce("Loading native model…")
    assert t.activity == AssistantActivity.IDLE_LISTEN
    assert t.bubble_state == "idle"


def test_user_presence_label_speaking_and_idle():
    assert user_presence_label(AssistantActivity.IDLE_LISTEN) == "Idle"
    assert user_presence_label(AssistantActivity.CAPTURING) == "Listening"
    assert user_presence_label(AssistantActivity.SPEAKING) == "Speaking"
    assert user_presence_label(AssistantActivity.WORKING, voice_output_muted=True) == "Writing"


def test_tray_and_menu_use_unified_labels():
    assert menu_status_line(AssistantActivity.IDLE_LISTEN) == "Idle"
    assert menu_status_line(AssistantActivity.CAPTURING) == "Listening"
    assert tray_tooltip_for_activity(AssistantActivity.CAPTURING) == "Qube — Listening"


def test_presence_service_emits_phase_for_transcribing():
    service = AssistantPresenceService()
    service.reduce("Transcribing...")
    snap = service.snapshot()
    assert snap.activity == AssistantActivity.WORKING
    assert snap.phase == AssistantPhase.STT


def test_phase_from_message_thinking_is_llm():
    phase = phase_from_message("Thinking...", AssistantActivity.WORKING, "thinking")
    assert phase == AssistantPhase.LLM


def test_companion_status_caption_maps_activity():
    assert companion_status_caption(AssistantActivity.WORKING, AssistantPhase.LLM) == "Thinking"
    assert (
        companion_status_caption(AssistantActivity.SPEAKING, AssistantPhase.TTS_STREAM)
        == "Speaking"
    )
    assert (
        companion_status_caption(AssistantActivity.CAPTURING, AssistantPhase.VAD_ACTIVE)
        == "Listening"
    )
    assert companion_status_caption(AssistantActivity.IDLE_LISTEN, None) is None


def test_presence_service_auto_caption_while_thinking():
    service = AssistantPresenceService()
    with patch("core.app_settings.get_companion_show_caption", return_value=True):
        service.reduce("Thinking...")
        assert service.snapshot().caption_text == "Thinking"


def test_presence_service_writing_caption_when_muted():
    service = AssistantPresenceService()
    service.set_voice_output_muted(True)
    with patch("core.app_settings.get_companion_show_caption", return_value=True):
        service.reduce("Thinking...")
        assert service.snapshot().caption_text == "Writing"


def test_presence_service_caption_roundtrip():
    service = AssistantPresenceService()
    service.set_caption_text("Hello world")
    assert service.snapshot().caption_text == "Hello world"


def test_ingestion_complete_must_force_idle_after_background_busy():
    """Regression: ingestion sets thinking + BACKGROUND_BUSY; completion must not stick."""
    reducer = AssistantActivityReducer()
    reducer.reduce("Ingesting Documents...")
    assert reducer.activity == AssistantActivity.BACKGROUND_BUSY
    assert reducer.bubble_state == "thinking"

    blocked = reducer.reduce("Indexed: notes.txt")
    assert blocked.blocked is True
    assert blocked.bubble_state == "thinking"

    reducer.set_background_busy(False)
    assert reducer.bubble_state == "thinking"
    assert reducer.activity == AssistantActivity.WORKING

    ok = reducer.reduce("Idle", force=True)
    assert ok.blocked is False
    assert ok.bubble_state == "idle"
    assert ok.activity == AssistantActivity.IDLE_LISTEN
    assert ok.display_text.strip() == "Idle"
