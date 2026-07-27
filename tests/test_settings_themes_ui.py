"""Settings → Themes UI tests (§14 Phase 4)."""

from __future__ import annotations

from core.theme.catalog import ThemeCatalog, catalog_for_registry
from core.theme.constants import UNRESOLVED_TOKEN_COLOR
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


def test_settings_themes_swatch_colors_sync_on_first_enter(main_window, qtbot):
    settings = main_window.ensure_settings_view()
    manager = main_window.theme_manager
    expected_accent = (
        manager.preview_resolve(scheme_id=manager.scheme_id)
        .core_tokens()
        .as_dict()["accent"]
    )

    settings._ensure_section_built("appearance.themes")
    accent_swatch = settings.themes_color_swatches["accent"]
    settings._themes_draft_controls_synced = False
    accent_swatch.set_color(UNRESOLVED_TOKEN_COLOR)

    settings.select_settings_section("appearance.themes")
    assert settings._themes_draft_controls_synced is True
    assert accent_swatch.color() == expected_accent
    assert not getattr(settings, "_themes_preview_initialized", False)


def test_settings_themes_section_builds(main_window, qtbot):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    qtbot.wait(100)

    assert hasattr(settings, "themes_theme_picker")
    assert getattr(settings, "_themes_preview_initialized", False)
    settings._ensure_themes_preview_initialized()
    qtbot.wait(100)
    settings._refresh_themes_preview()
    qtbot.wait(100)
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


def test_settings_themes_conversations_preview_paints_on_section_enter(main_window, qtbot):
    """Chat wallpaper preview must render on first enter without manual refresh."""
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    qtbot.wait(500)

    assert getattr(settings, "_themes_preview_initialized", False)
    panel = settings.themes_preview_panel
    pixmap = panel._conversations_view.grab()
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0


def test_settings_themes_preview_panels_have_visible_height(main_window, qtbot):
    """Preview shells must reserve snapshot height (not collapse to a sliver)."""
    settings = main_window.ensure_settings_view()
    settings._ensure_section_built("appearance.themes")
    settings.select_settings_section("appearance.themes")
    qtbot.wait(500)

    for attr in (
        "themes_preview_panel",
        "themes_components_preview_panel",
        "themes_library_preview_panel",
    ):
        panel = getattr(settings, attr)
        assert panel is not None
        assert panel.height() >= 200


def test_settings_themes_preview_snapshot_updates_on_draft_color_change(main_window, qtbot):
    """Theme color draft edits must repaint the components preview snapshot."""
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    qtbot.wait(500)
    settings._ensure_themes_previews_initialized()
    settings._refresh_themes_preview()
    qtbot.wait(200)

    comp = settings.themes_components_preview_panel
    shell = comp._components_live._shell

    before_theme = settings._draft_resolved_theme()
    assert before_theme is not None

    settings._on_themes_color_changed("background", "#001122")
    settings._refresh_themes_preview()
    qtbot.wait(200)

    after_theme = settings._draft_resolved_theme()
    assert after_theme is not None
    assert after_theme.background.lower() == "#001122"
    assert after_theme.background.lower() != before_theme.background.lower()

    shell_stylesheet = shell.styleSheet().lower()
    assert "#001122" in shell_stylesheet


def test_settings_themes_draft_preview_uses_scheme_only(main_window, qtbot):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_previews_initialized()
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
    assert settings.themes_components_preview_panel._components_live._primary_btn is not None
