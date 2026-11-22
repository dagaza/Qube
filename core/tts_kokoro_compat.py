"""Compatibility shims for kokoro-onnx vs bundled kokoro-v1.0.onnx."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("Qube.Audio")

_MIN_KOKORO_ONNX = "0.6.0"


def kokoro_onnx_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("kokoro-onnx")
    except Exception:
        return None


def kokoro_onnx_needs_speed_dtype_patch() -> bool:
    ver = kokoro_onnx_version()
    if not ver:
        return False
    try:
        from packaging.version import Version

        return Version(ver) < Version(_MIN_KOKORO_ONNX)
    except Exception:
        return False


def patch_kokoro_engine_session_run(engine: Any) -> None:
    """
    kokoro-onnx <0.6.0 feeds speed as int32 on graphs that expect float32 (input_ids export).
    Coerce speed inputs at the ONNX session boundary so synthesis still works on older pins.
    """
    if not kokoro_onnx_needs_speed_dtype_patch():
        return
    sess = getattr(engine, "sess", None)
    if sess is None or getattr(sess, "_qube_speed_dtype_patch", False):
        return
    original_run = sess.run

    def run_patched(output_names, input_feed, run_options=None):
        if isinstance(input_feed, dict) and "speed" in input_feed:
            speed_arr = input_feed["speed"]
            if isinstance(speed_arr, np.ndarray) and speed_arr.dtype != np.float32:
                input_feed = dict(input_feed)
                input_feed["speed"] = np.asarray(speed_arr, dtype=np.float32)
        return original_run(output_names, input_feed, run_options)

    sess.run = run_patched  # type: ignore[method-assign]
    sess._qube_speed_dtype_patch = True
    logger.info(
        "Applied kokoro-onnx speed dtype patch (installed %s; recommend >= %s).",
        kokoro_onnx_version(),
        _MIN_KOKORO_ONNX,
    )