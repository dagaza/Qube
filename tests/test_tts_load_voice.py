"""TTSWorker.load_voice success/failure contract."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from workers.tts_worker import TTSWorker


def test_load_voice_rejects_unsupported_and_keeps_prior_adapter():
    worker = TTSWorker.__new__(TTSWorker)
    worker.audio = MagicMock()
    worker.status_update = MagicMock()
    worker.model_loaded = MagicMock()
    worker.active_adapter = object()
    worker.model_path = "/old/model.onnx"
    worker.active_voice_name = "af_sarah"
    worker.stream = None
    worker.current_device_index = None

    ok = TTSWorker.load_voice(worker, "/tmp/unknown-tts.onnx")

    assert ok is False
    assert worker.active_adapter is not None
    worker.status_update.emit.assert_called_once()
    worker.model_loaded.emit.assert_not_called()


def test_load_voice_rolls_back_adapter_on_init_failure():
    worker = TTSWorker.__new__(TTSWorker)
    worker.audio = MagicMock()
    worker.status_update = MagicMock()
    worker.model_loaded = MagicMock()
    prior = object()
    worker.active_adapter = prior
    worker.model_path = "/old/kokoro-v1.0.onnx"
    worker.active_voice_name = "af_sarah"
    worker.stream = None
    worker.current_device_index = None

    with patch("workers.tts_worker.KokoroAdapter", side_effect=RuntimeError("bad onnx")):
        ok = TTSWorker.load_voice(worker, "/tmp/kokoro-v1.0.onnx")

    assert ok is False
    assert worker.active_adapter is prior
    assert worker.model_path == "/old/kokoro-v1.0.onnx"
    worker.model_loaded.emit.assert_not_called()
