"""Settings → Themes UI tests (§14 Phase 4)."""

from __future__ import annotations

from core.theme.catalog import ThemeCatalog, catalog_for_registry
from core.theme.schemes import (
    BUILTIN_SCHEMES,
    BUILTIN_CATPUCCIN_LATTE_ID,
    DEFAULT_SCHEME_ID_DARK,
    DEFAULT_SCHEME_ID_LIGHT,
)
from core.theme.tokens import ThemeMode


def test_settings_themes_display_names_use_catalog():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    assert catalog.display_name(DEFAULT_SCHEME_ID_DARK) == "Catppuccin Dark"
    assert catalog.display_name(BUILTIN_CATPUCCIN_LATTE_ID) == "Catppuccin Light"


def test_settings_themes_paired_family_has_two_variant_members():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    members = catalog.members_of_family("catppuccin")
    assert DEFAULT_SCHEME_ID_DARK in members
    assert BUILTIN_CATPUCCIN_LATTE_ID in members
    polarities = {catalog.get_definition(member).base_mode for member in members}
    assert polarities == {"dark", "light"}


def test_settings_themes_dracula_family_needs_light_fallback():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    sibling = catalog.sibling_for_polarity("builtin.dracula", ThemeMode.LIGHT)
    assert sibling is None
    assert catalog.fallback_for_family("dracula", ThemeMode.LIGHT) == DEFAULT_SCHEME_ID_LIGHT


def test_settings_themes_picker_model_includes_all_builtins():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    model = catalog.themes_for_picker()
    assert len(model.entries) == len(BUILTIN_SCHEMES)


def test_settings_themes_section_builds(main_window, qtbot):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()
    qtbot.wait(10)

    assert hasattr(settings, "themes_theme_picker")
    assert not hasattr(settings, "themes_mode_card")
    catalog = ThemeCatalog(main_window.theme_manager.list_schemes())
    assert (
        settings.themes_theme_picker.text()
        == catalog.display_name(main_window.theme_manager.scheme_id)
    )

    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    settings._rebuild_variant_row(catalog, DEFAULT_SCHEME_ID_DARK)
    assert len(settings.themes_variant_cbs) == 2

    settings._rebuild_variant_row(catalog, "builtin.dracula")
    assert not settings.themes_unavailable_row.isHidden()
    assert "light variant" in settings.themes_unavailable_label.text().lower()


def test_settings_themes_draft_preview_uses_scheme_only(main_window, qtbot):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()
    qtbot.wait(10)

    manager = main_window.theme_manager
    applied_before = manager.current

    class RecordingApplicator:
        apply_count = 0

        def apply(self, resolved, *, profiler=None):
            RecordingApplicator.apply_count += 1

    recording = RecordingApplicator()
    manager._applicator = recording  # type: ignore[assignment]

    settings._select_themes_scheme(BUILTIN_CATPUCCIN_LATTE_ID)
    settings._refresh_themes_preview()

    assert RecordingApplicator.apply_count == 0
    assert manager.current.scheme_id == applied_before.scheme_id
    assert settings.themes_preview_panel._components_live._primary_btn is not None
