"""kokoro-onnx speed input compatibility."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from core.tts_kokoro_compat import patch_kokoro_engine_session_run


def test_patch_coerces_speed_to_float32_before_sess_run(monkeypatch):
    calls: list[dict] = []

    def fake_version(_name: str) -> str:
        return "0.5.0"

    monkeypatch.setattr(
        "core.tts_kokoro_compat.kokoro_onnx_version",
        lambda: "0.5.0",
    )

    sess = MagicMock()
    sess._qube_speed_dtype_patch = False

    def record_run(_output_names, input_feed, run_options=None):
        calls.append(dict(input_feed))
        return [np.zeros(4, dtype=np.float32)]

    sess.run = record_run
    engine = MagicMock(sess=sess)

    patch_kokoro_engine_session_run(engine)

    feed = {
        "input_ids": np.zeros((1, 3), dtype=np.int64),
        "style": np.zeros(4, dtype=np.float32),
        "speed": np.array([1], dtype=np.int32),
    }
    engine.sess.run(None, feed)

    assert calls
    assert calls[0]["speed"].dtype == np.float32