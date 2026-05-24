"""Tests for assistant activity reducer."""

from core.assistant_activity import AssistantActivity, AssistantActivityReducer
from core.assistant_presence import AssistantPresenceService, AssistantPhase, companion_status_caption, phase_from_message


def test_reducer_blocks_stray_idle_during_recording():
    reducer = AssistantActivityReducer()
    reducer.reduce("🎙️ RECORDING...")
    blocked = reducer.reduce("Idle")
    assert blocked.blocked is True
    assert blocked.bubble_state == "recording"


def test_reducer_allows_voice_capture_idle():
    reducer = AssistantActivityReducer()
    reducer.reduce("🎙️ RECORDING...")
    ok = reducer.reduce("Voice capture idle")
    assert ok.blocked is False
    assert ok.bubble_state == "idle"


def test_reducer_maps_thinking_to_working_activity():
    reducer = AssistantActivityReducer()
    t = reducer.reduce("Thinking...")
    assert t.activity == AssistantActivity.WORKING
    assert t.bubble_state == "thinking"


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


def test_reducer_voice_paused_forces_assistant_off():
    reducer = AssistantActivityReducer()
    reducer.set_voice_paused(True)
    t = reducer.reduce("Idle")
    assert t.activity == AssistantActivity.ASSISTANT_OFF


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
    assert companion_status_caption(AssistantActivity.WORKING, AssistantPhase.LLM) == "Thinking…"
    assert companion_status_caption(AssistantActivity.SPEAKING, AssistantPhase.TTS_STREAM) == "Speaking…"
    assert companion_status_caption(AssistantActivity.CAPTURING, AssistantPhase.VAD_ACTIVE) == "Listening…"
    assert companion_status_caption(AssistantActivity.IDLE_LISTEN, None) is None


def test_presence_service_auto_caption_while_thinking():
    service = AssistantPresenceService()
    service.reduce("Thinking...")
    assert service.snapshot().caption_text == "Thinking…"


def test_presence_service_caption_roundtrip():
    service = AssistantPresenceService()
    service.set_caption_text("Hello world")
    assert service.snapshot().caption_text == "Hello world"
