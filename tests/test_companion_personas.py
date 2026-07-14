"""Tests for companion persona registry and settings."""

from __future__ import annotations

from core.companion_cube_style import CompanionCubeStyle, DEFAULT_COMPANION_CUBE_STYLE, normalize_companion_cube_style
from core.companion_personas import (
    CompanionPersonaId,
    DEFAULT_COMPANION_PERSONA,
    normalize_companion_persona,
)
from ui.companion.personas.base import get_persona_renderer
from ui.companion.personas.qube_cube_classic import QubeCubeClassicPersonaRenderer
from ui.companion.personas.qube_cube_experimental import QubeCubeExperimentalPersonaRenderer
from ui.companion.personas.sphere import SpherePersonaRenderer


def test_normalize_companion_persona_defaults_unknown():
    assert normalize_companion_persona(None) == DEFAULT_COMPANION_PERSONA
    assert normalize_companion_persona("") == DEFAULT_COMPANION_PERSONA
    assert normalize_companion_persona("invalid") == DEFAULT_COMPANION_PERSONA


def test_normalize_companion_persona_accepts_values():
    assert normalize_companion_persona("sphere") == CompanionPersonaId.SPHERE
    assert normalize_companion_persona("QUBE") == CompanionPersonaId.QUBE
    assert normalize_companion_persona(CompanionPersonaId.QUBE) == CompanionPersonaId.QUBE


def test_get_persona_renderer_returns_distinct_renderers():
    sphere = get_persona_renderer(CompanionPersonaId.SPHERE)
    qube = get_persona_renderer(CompanionPersonaId.QUBE)
    assert isinstance(sphere, SpherePersonaRenderer)
    assert isinstance(qube, (QubeCubeClassicPersonaRenderer, QubeCubeExperimentalPersonaRenderer))
    assert sphere.persona_id == CompanionPersonaId.SPHERE
    assert qube.persona_id == CompanionPersonaId.QUBE


def test_get_persona_renderer_respects_cube_style(monkeypatch):
    store: dict[str, object] = {}

    class FakeStore:
        def get(self, key, default=None):
            return store.get(key, default)

        def set(self, key, value):
            store[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: FakeStore())

    from core import app_settings

    app_settings.set_companion_cube_style("classic")
    assert isinstance(get_persona_renderer(CompanionPersonaId.QUBE), QubeCubeClassicPersonaRenderer)
    app_settings.set_companion_cube_style("experimental")
    assert isinstance(get_persona_renderer(CompanionPersonaId.QUBE), QubeCubeExperimentalPersonaRenderer)


def test_normalize_companion_cube_style_defaults_unknown():
    assert normalize_companion_cube_style(None) == DEFAULT_COMPANION_CUBE_STYLE
    assert normalize_companion_cube_style("") == DEFAULT_COMPANION_CUBE_STYLE
    assert normalize_companion_cube_style("invalid") == DEFAULT_COMPANION_CUBE_STYLE
    assert normalize_companion_cube_style("classic") == CompanionCubeStyle.CLASSIC
    assert normalize_companion_cube_style("EXPERIMENTAL") == CompanionCubeStyle.EXPERIMENTAL


def test_companion_cube_style_settings_round_trip(monkeypatch):
    store: dict[str, object] = {}

    class FakeStore:
        def get(self, key, default=None):
            return store.get(key, default)

        def set(self, key, value):
            store[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: FakeStore())

    from core import app_settings

    assert app_settings.get_companion_cube_style() == CompanionCubeStyle.CLASSIC
    app_settings.set_companion_cube_style("classic")
    assert app_settings.get_companion_cube_style() == CompanionCubeStyle.CLASSIC
    app_settings.set_companion_cube_style("experimental")
    assert app_settings.get_companion_cube_style() == CompanionCubeStyle.EXPERIMENTAL


def test_companion_persona_settings_round_trip(monkeypatch):
    store: dict[str, object] = {}

    class FakeStore:
        def get(self, key, default=None):
            return store.get(key, default)

        def set(self, key, value):
            store[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: FakeStore())

    from core import app_settings

    assert app_settings.get_companion_persona() == CompanionPersonaId.QUBE
    app_settings.set_companion_persona("sphere")
    assert app_settings.get_companion_persona() == CompanionPersonaId.SPHERE
    app_settings.set_companion_persona("qube")
    assert app_settings.get_companion_persona() == CompanionPersonaId.QUBE
