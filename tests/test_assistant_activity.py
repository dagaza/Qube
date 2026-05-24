"""Tests for assistant activity reducer."""

from core.assistant_activity import AssistantActivity, AssistantActivityReducer


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
