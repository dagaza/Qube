"""Tests for theme customization identity (§14 Phase 5)."""

from __future__ import annotations

from core.theme.catalog import catalog_for_registry
from core.theme.customization_identity import (
    customization_identity_text,
    customization_is_active,
    lineage_root_scheme_id,
    suggested_custom_theme_name,
)
from core.theme.definition import ColorSchemeDefinition
from core.theme.io import export_color_scheme
from core.theme.schemes import BUILTIN_SCHEMES, DEFAULT_SCHEME_ID_DARK
from core.theme.manager import ThemeManager
from core.theme.storage import ThemeStorage
from core.theme.tokens import ThemeMode


def test_customization_is_active():
    assert customization_is_active({}) is False
    assert customization_is_active({"accent": "#ff0000"}) is True


def test_identity_preset_without_overrides():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    text = customization_identity_text(
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        overrides={},
        catalog=catalog,
    )
    assert text == "Based on: Catppuccin Dark"


def test_identity_preset_with_overrides():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    text = customization_identity_text(
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        overrides={"accent": "#ff0000"},
        catalog=catalog,
    )
    assert text == "Custom · based on Catppuccin Dark"


def test_identity_saved_custom_theme():
    registry = {
        **BUILTIN_SCHEMES,
        "user.my-catppuccin": ColorSchemeDefinition(
            id="user.my-catppuccin",
            name="My Catppuccin",
            base_mode="dark",
            family="catppuccin",
            extends=DEFAULT_SCHEME_ID_DARK,
            algorithm="catppuccin",
            overrides={"accent": "#010203"},
        ),
    }
    catalog = catalog_for_registry(registry)
    text = customization_identity_text(
        scheme_id="user.my-catppuccin",
        overrides={},
        catalog=catalog,
    )
    assert text == "My Catppuccin · based on Catppuccin Dark"


def test_identity_saved_custom_with_unsaved_edits():
    registry = {
        **BUILTIN_SCHEMES,
        "user.my-catppuccin": ColorSchemeDefinition(
            id="user.my-catppuccin",
            name="My Catppuccin",
            base_mode="dark",
            family="catppuccin",
            extends=DEFAULT_SCHEME_ID_DARK,
            algorithm="catppuccin",
            overrides={"accent": "#010203"},
        ),
    }
    catalog = catalog_for_registry(registry)
    text = customization_identity_text(
        scheme_id="user.my-catppuccin",
        overrides={"accent": "#020406"},
        catalog=catalog,
    )
    assert text == "Custom · My Catppuccin (unsaved changes)"


def test_suggested_custom_theme_name():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    assert suggested_custom_theme_name(DEFAULT_SCHEME_ID_DARK, catalog) == (
        "My Catppuccin Dark"
    )


def test_lineage_root_from_custom_theme():
    registry = {
        **BUILTIN_SCHEMES,
        "user.my-catppuccin": ColorSchemeDefinition(
            id="user.my-catppuccin",
            name="My Catppuccin",
            base_mode="dark",
            family="catppuccin",
            extends=DEFAULT_SCHEME_ID_DARK,
            algorithm="catppuccin",
            overrides={"accent": "#010203"},
        ),
    }
    catalog = catalog_for_registry(registry)
    assert lineage_root_scheme_id("user.my-catppuccin", catalog) == DEFAULT_SCHEME_ID_DARK


def test_export_includes_family_metadata():
    definition = BUILTIN_SCHEMES[DEFAULT_SCHEME_ID_DARK]
    payload = export_color_scheme(definition)
    assert payload["schema"] == 2
    assert payload["family"] == "catppuccin"
    assert payload["variant"] == "mocha"


def test_save_draft_persists_family(tmp_path, monkeypatch):
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: tmp_path)
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    definition = manager.save_draft_as_custom_scheme(
        name="My Catppuccin",
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        overrides={"accent": "#ff0000"},
    )
    assert definition.family == "catppuccin"
    assert definition.extends == DEFAULT_SCHEME_ID_DARK
    assert definition.variant is None

    payload = export_color_scheme(definition)
    assert payload["family"] == "catppuccin"
    assert payload["extends"] == DEFAULT_SCHEME_ID_DARK
