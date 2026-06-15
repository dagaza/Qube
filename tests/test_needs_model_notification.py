"""Tests for the needs-model in-app notification."""

from __future__ import annotations

from core.notification_types import NotificationSeverity, needs_model_event


def test_needs_model_event_without_local_models(monkeypatch):
    monkeypatch.setattr("core.app_settings.get_engine_mode", lambda: "internal")
    monkeypatch.setattr(
        "core.local_gguf_library.has_local_gguf_models", lambda: False
    )

    event = needs_model_event()
    assert event.action_id == "open_models"
    assert event.action_label == "Open Models"
    assert event.auto_dismiss_ms == 5000
    assert event.show_countdown is True
    req = event.to_app_request()
    assert req.show_countdown is True
    assert req.auto_dismiss_ms == 5000


def test_needs_model_event_with_local_models(monkeypatch):
    monkeypatch.setattr("core.app_settings.get_engine_mode", lambda: "internal")
    monkeypatch.setattr("core.local_gguf_library.has_local_gguf_models", lambda: True)

    event = needs_model_event()
    assert event.action_id == "open_local_model_picker"
    assert event.action_label == "Select AI Model"
    assert event.auto_dismiss_ms == 5000
    assert event.show_countdown is True
    assert event.severity == NotificationSeverity.CRITICAL


def test_needs_model_event_external_engine_uses_model_manager(monkeypatch):
    monkeypatch.setattr("core.app_settings.get_engine_mode", lambda: "external")
    monkeypatch.setattr("core.local_gguf_library.has_local_gguf_models", lambda: True)

    event = needs_model_event()
    assert event.action_id == "open_models"


def test_needs_model_event_is_not_rate_limited(monkeypatch):
    monkeypatch.setattr("core.app_settings.get_engine_mode", lambda: "internal")
    monkeypatch.setattr("core.local_gguf_library.has_local_gguf_models", lambda: False)

    event = needs_model_event()
    assert event.rate_limit_key is None
    assert event.rate_limit_sec == 0.0

    from core.notification_service import NotificationService

    delivered: list = []
    svc = NotificationService()
    svc.set_window_state_providers(visible=lambda: True, focused=lambda: True)
    svc.set_show_handlers(in_app=lambda e: delivered.append(e))

    svc.emit(event)
    svc.emit(needs_model_event())
    assert len(delivered) == 2
