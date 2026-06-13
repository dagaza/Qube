"""Rotating sample phrases for TTS voice preview in Settings."""

from __future__ import annotations

TTS_VOICE_PREVIEW_PHRASES: tuple[str, ...] = (
    "Hi! If you like my voice, I'd be happy to accompany you as your assistant.",
    "Hello there — pick the voice that feels right, and we'll get started together.",
    "Testing one, two, three. Ready to help whenever you are.",
    "A warm welcome! I'm here to make every answer sound a little more human.",
    "Sound check complete. Let's make your conversations feel effortless.",
)

_PREVIEW_SESSION_ID = "__voice_preview__"


def next_tts_voice_preview_phrase(index: int) -> tuple[str, int]:
    """Return the phrase at ``index`` and the next index for rotation."""
    phrases = TTS_VOICE_PREVIEW_PHRASES
    if not phrases:
        return "", index
    return phrases[index % len(phrases)], index + 1


def is_voice_preview_session(session_id: str) -> bool:
    return session_id == _PREVIEW_SESSION_ID
