"""Tests for UI language variant settings and localisation."""

from __future__ import annotations

from core.ui_language import (
    DEFAULT_UI_LANGUAGE,
    UiLanguage,
    localize_text,
    normalize_ui_language,
    tr,
)


def test_normalize_ui_language_defaults_unknown():
    assert normalize_ui_language(None) == DEFAULT_UI_LANGUAGE
    assert normalize_ui_language("") == DEFAULT_UI_LANGUAGE
    assert normalize_ui_language("invalid") == DEFAULT_UI_LANGUAGE


def test_normalize_ui_language_accepts_values():
    assert normalize_ui_language("british") == UiLanguage.BRITISH
    assert normalize_ui_language("AMERICAN") == UiLanguage.AMERICAN
    assert normalize_ui_language(UiLanguage.AMERICAN) == UiLanguage.AMERICAN


def test_localize_text_british_is_canonical():
    text = "Minimise window and check colour behaviour."
    assert localize_text(text, UiLanguage.BRITISH) == text
    assert tr(text, UiLanguage.BRITISH) == text


def test_localize_text_american_replaces_spellings():
    text = "Minimise window; grey status; optimised behaviour and colour."
    result = localize_text(text, UiLanguage.AMERICAN)
    assert "Minimize" in result
    assert "gray" in result
    assert "optimized" in result
    assert "behavior" in result
    assert "color" in result
    assert "Minimise" not in result
    assert "grey" not in result


def test_ui_language_settings_round_trip(monkeypatch):
    store: dict[str, object] = {}

    class FakeStore:
        def get(self, key, default=None):
            return store.get(key, default)

        def set(self, key, value):
            store[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: FakeStore())

    from core import app_settings

    assert app_settings.get_ui_language() == UiLanguage.BRITISH
    app_settings.set_ui_language("american")
    assert app_settings.get_ui_language() == UiLanguage.AMERICAN
    app_settings.set_ui_language("british")
    assert app_settings.get_ui_language() == UiLanguage.BRITISH
