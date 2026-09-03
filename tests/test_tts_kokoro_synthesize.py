"""Kokoro TTS synthesis and output-stream fallback behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from workers.tts_worker import KokoroAdapter, TTSWorker


def test_kokoro_synthesize_uses_sync_create():
    adapter = KokoroAdapter.__new__(KokoroAdapter)
    adapter.sample_rate = 24000
    adapter.engine = MagicMock()
    adapter.engine.create.return_value = (np.zeros(2400, dtype=np.float32), 24000)

    chunks = list(KokoroAdapter.synthesize(adapter, "Hello preview.", "af_heart"))

    adapter.engine.create.assert_called_once_with(
        "Hello preview.",
        voice="af_heart",
        speed=1.0,
        lang="en-us",
    )
    assert chunks
    assert all(isinstance(chunk, bytes) and chunk for chunk in chunks)


def test_open_output_stream_falls_back_to_default_device():
    worker = TTSWorker.__new__(TTSWorker)
    worker.audio = MagicMock()
    worker.current_device_index = 14
    good_stream = MagicMock()
    worker.audio.open.side_effect = [OSError("invalid device"), good_stream]

    stream = TTSWorker._open_output_stream(worker, 24000)

    assert stream is good_stream
    assert worker.current_device_index is None
    assert worker.audio.open.call_count == 2
