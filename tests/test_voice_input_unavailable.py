from core.notification_types import (
    VOICE_INPUT_UNAVAILABLE_BODY,
    voice_input_unavailable_event,
)


def test_voice_input_unavailable_event_copy() -> None:
    event = voice_input_unavailable_event()
    assert event.title == "Voice input unavailable"
    assert event.body == VOICE_INPUT_UNAVAILABLE_BODY
    assert "another application may be using your microphone" in event.body
    assert "text-based mode" in event.body
    assert event.dedupe_key == "voice_input_unavailable"
    assert event.action_id is None
