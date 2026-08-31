"""Tests for native model hardware-reload notifications."""

from __future__ import annotations

from core.notification_types import native_model_reloaded_from_settings_event


def test_native_model_reloaded_event_default_body():
    event = native_model_reloaded_from_settings_event(model_name="Qwen3-8B-Q4_K_M.gguf")
    assert event.title == "Model reloaded"
    assert "updated hardware settings" in event.body
    assert event.rate_limit_key == "native_hardware_reload"


def test_native_model_reloaded_event_cpu_fallback_body():
    event = native_model_reloaded_from_settings_event(
        model_name="Qwen3-8B-Q4_K_M.gguf",
        cpu_fallback=True,
    )
    assert "ready on CPU" in event.body
    assert "GPU layers" in event.body
