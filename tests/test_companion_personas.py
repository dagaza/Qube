"""Tests for companion persona registry and settings."""

from __future__ import annotations

from core.companion_personas import (
    CompanionPersonaId,
    DEFAULT_COMPANION_PERSONA,
    normalize_companion_persona,
)
from ui.companion.personas.base import get_persona_renderer
from ui.companion.personas.qube_cube import QubeCubePersonaRenderer
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
    assert isinstance(qube, QubeCubePersonaRenderer)
    assert sphere.persona_id == CompanionPersonaId.SPHERE
    assert qube.persona_id == CompanionPersonaId.QUBE


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
