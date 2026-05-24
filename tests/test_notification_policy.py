"""Tests for notification policy."""

from core import app_settings
from core.notification_policy import plan_delivery
from core.notification_types import NotificationEvent, NotificationSeverity


def test_focused_suppresses_success(monkeypatch):
    monkeypatch.setattr(app_settings, "get_notifications_enabled", lambda: True)
    monkeypatch.setattr(app_settings, "get_notifications_dnd", lambda: False)
    monkeypatch.setattr(app_settings, "get_notifications_suppress_when_focused", lambda: True)
    event = NotificationEvent(
        title="Reply ready",
        body="Done",
        severity=NotificationSeverity.SUCCESS,
        category="turn",
    )
    plan = plan_delivery(event, window_visible=True, window_focused=True)
    assert plan.show_in_app is False
    assert plan.show_os is False


def test_hidden_delivers_os_for_success(monkeypatch):
    monkeypatch.setattr(app_settings, "get_notifications_enabled", lambda: True)
    monkeypatch.setattr(app_settings, "get_notifications_dnd", lambda: False)
    monkeypatch.setattr(app_settings, "get_notifications_os_when_hidden", lambda: True)
    monkeypatch.setattr(app_settings, "get_notifications_suppress_when_focused", lambda: True)
    event = NotificationEvent(
        title="Reply ready",
        body="Done",
        severity=NotificationSeverity.SUCCESS,
        category="turn",
    )
    plan = plan_delivery(event, window_visible=False, window_focused=False)
    assert plan.show_os is True
