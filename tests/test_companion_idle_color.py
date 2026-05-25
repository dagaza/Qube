"""Tests for companion idle color presets and settings."""

from __future__ import annotations

from core.assistant_activity import AssistantActivity
from core.companion_idle_color import (
    CompanionIdleColor,
    DEFAULT_COMPANION_IDLE_COLOR,
    idle_color_pair,
    normalize_companion_idle_color,
)
from ui.companion.personas.colors import ACTIVITY_COLORS, activity_color_pair


def test_normalize_companion_idle_color_defaults_unknown():
    assert normalize_companion_idle_color(None) == DEFAULT_COMPANION_IDLE_COLOR
    assert normalize_companion_idle_color("") == DEFAULT_COMPANION_IDLE_COLOR
    assert normalize_companion_idle_color("invalid") == DEFAULT_COMPANION_IDLE_COLOR


def test_normalize_companion_idle_color_accepts_values():
    assert normalize_companion_idle_color("purple") == CompanionIdleColor.PURPLE
    assert normalize_companion_idle_color("BLUE") == CompanionIdleColor.BLUE
    assert normalize_companion_idle_color(CompanionIdleColor.PURPLE) == CompanionIdleColor.PURPLE


def test_idle_color_pair_presets():
    assert idle_color_pair(CompanionIdleColor.PURPLE) == ("#8b5cf6", "#a78bfa")
    assert idle_color_pair(CompanionIdleColor.BLUE) == ("#89b4fa", "#b4befe")


def test_activity_color_pair_idle_respects_preset():
    purple = activity_color_pair(AssistantActivity.IDLE_LISTEN, CompanionIdleColor.PURPLE)
    blue = activity_color_pair(AssistantActivity.IDLE_LISTEN, CompanionIdleColor.BLUE)
    assert purple == ("#8b5cf6", "#a78bfa")
    assert blue == ("#89b4fa", "#b4befe")


def test_activity_color_pair_non_idle_unchanged():
    working = activity_color_pair(AssistantActivity.WORKING, CompanionIdleColor.PURPLE)
    assert working == ACTIVITY_COLORS[AssistantActivity.WORKING]


def test_companion_idle_color_settings_round_trip(monkeypatch):
    store: dict[str, object] = {}

    class FakeStore:
        def get(self, key, default=None):
            return store.get(key, default)

        def set(self, key, value):
            store[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: FakeStore())

    from core import app_settings

    assert app_settings.get_companion_idle_color() == CompanionIdleColor.PURPLE
    app_settings.set_companion_idle_color("blue")
    assert app_settings.get_companion_idle_color() == CompanionIdleColor.BLUE
