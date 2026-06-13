"""TTS voice preview helpers and queue behavior."""


def test_preview_phrases_rotate():
    from core.tts_voice_preview import (
        TTS_VOICE_PREVIEW_PHRASES,
        next_tts_voice_preview_phrase,
    )

    first, idx = next_tts_voice_preview_phrase(0)
    second, idx = next_tts_voice_preview_phrase(idx)
    wrapped, _ = next_tts_voice_preview_phrase(
        len(TTS_VOICE_PREVIEW_PHRASES)
    )

    assert first == TTS_VOICE_PREVIEW_PHRASES[0]
    assert second == TTS_VOICE_PREVIEW_PHRASES[1]
    assert wrapped == TTS_VOICE_PREVIEW_PHRASES[0]


def test_queue_voice_preview_uses_preview_sentinel():
    from workers.tts_worker import TTSWorker, _VOICE_PREVIEW

    worker = TTSWorker.__new__(TTSWorker)
    worker.sentence_queue = __import__("queue").Queue()
    worker._last_queued_tts_key = ""
    worker._interrupt_tts = False
    worker.active_adapter = object()
    worker.isRunning = lambda: False
    worker.start = lambda: None

    worker.queue_voice_preview("Hello preview.")

    item = worker.sentence_queue.get()
    assert item == (_VOICE_PREVIEW, "Hello preview.")
