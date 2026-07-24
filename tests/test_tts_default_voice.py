"""Default Kokoro TTS voice selection."""

from __future__ import annotations

from core.tts_models import DEFAULT_KOKORO_VOICE, resolve_default_tts_voice


def test_resolve_default_tts_voice_prefers_af_heart():
    voices = ["af_bella", "af_heart", "am_adam"]
    assert resolve_default_tts_voice(voices) == "af_heart"


def test_resolve_default_tts_voice_falls_back_when_missing():
    assert resolve_default_tts_voice(["af_bella", "am_adam"]) == "af_bella"
    assert resolve_default_tts_voice(["Default"]) == "Default"
    assert resolve_default_tts_voice([]) == DEFAULT_KOKORO_VOICE


def test_load_voice_selects_af_heart_for_kokoro():
    from unittest.mock import MagicMock, patch

    from workers.tts_worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker.audio = MagicMock()
    worker.status_update = MagicMock()
    worker.model_loaded = MagicMock()
    worker.active_adapter = None
    worker.model_path = ""
    worker.active_voice_name = "Default"
    worker.stream = None
    worker.current_device_index = None

    adapter = MagicMock()
    adapter.sample_rate = 24000
    adapter.available_voices = ["af_bella", "af_heart", "am_adam"]
    worker.audio.open.return_value = MagicMock()

    with patch("core.tts_models.classify_tts_architecture", return_value="kokoro"), patch(
        "workers.tts_worker.KokoroAdapter", return_value=adapter
    ):
        ok = TTSWorker.load_voice(worker, "/tmp/kokoro-v1.0.onnx")

    assert ok is True
    assert worker.active_voice_name == "af_heart"
