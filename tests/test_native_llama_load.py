"""Tests for native GGUF load fallback planning."""

from __future__ import annotations

from core.gpu_layers_cap import (
    default_internal_n_gpu_layers_suggested,
    max_safe_n_gpu_layers,
    reset_gpu_vram_cache_for_tests,
)
from core.native_llama_load import is_retryable_native_load_error, native_load_attempts
from core.native_load_errors import format_native_load_failure_dialog


def test_llama_context_error_is_retryable():
    assert is_retryable_native_load_error(ValueError("Failed to create llama_context"))


def test_corrupt_file_error_is_not_retryable():
    assert not is_retryable_native_load_error(
        ValueError("Failed to load model from file: /tmp/bad.gguf")
    )


def test_native_load_attempts_includes_cpu_fallback():
    plans = native_load_attempts(33, 4096)
    assert plans[0] == (33, 4096, "requested")
    assert (0, 4096, "cpu") in plans


def test_llama_context_dialog_mentions_gpu_layers():
    _title, body = format_native_load_failure_dialog(
        error="Failed to create llama_context",
    )
    assert "GPU layers" in body


def test_win32_unknown_vram_defaults_cpu_only(monkeypatch):
    reset_gpu_vram_cache_for_tests()
    monkeypatch.setattr("core.gpu_layers_cap.sys.platform", "win32")
    monkeypatch.setattr("core.gpu_layers_cap.detect_gpu_vram_bytes", lambda: 0)
    assert max_safe_n_gpu_layers() == 32
    assert default_internal_n_gpu_layers_suggested() == 0
