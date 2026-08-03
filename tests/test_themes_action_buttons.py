"""Themes Revert/Cancel/Apply brand button styling and draft-aware enabled state."""

from __future__ import annotations

import re

from PyQt6.QtGui import QIcon

from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT, SURFACE_LIBRARY_PREVIEW
from core.surface_fill.models import SurfaceProfile, WallpaperNone
from ui.components.brand_buttons import BRAND_CAUTION, BRAND_DANGER, brand_label_color


def _qss_label_color(style_sheet: str) -> str | None:
    match = re.search(r"(?<![-\w])color:\s*([^;!]+)", style_sheet)
    return match.group(1).strip() if match else None


def _icon_has_light_pixels(
    icon: QIcon, *, size: int = 24, mode: QIcon.Mode = QIcon.Mode.Normal
) -> bool:
    pix = icon.pixmap(size, size, mode, QIcon.State.Off)
    img = pix.toImage()
    for y in range(img.height()):
        for x in range(img.width()):
            color = img.pixelColor(x, y)
            if color.alpha() > 64 and color.lightness() > 180:
                return True
    return False


def test_themes_action_buttons_stay_branded(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()

    for name, object_name in (
        ("themes_reset_btn", "ThemesResetButton"),
        ("themes_revert_btn", "ThemesRevertButton"),
        ("themes_cancel_btn", "ThemesCancelButton"),
        ("themes_apply_btn", "ThemesApplyButton"),
    ):
        btn = getattr(settings, name)
        assert btn is not None
        assert btn.objectName() == object_name
        ss = btn.styleSheet()
        assert len(ss) > 100, f"{name} missing widget-level brand QSS"
        assert f"#{object_name}" in ss, f"{name} QSS should scope to object name"
        class_tag = str(btn.property("class"))
        if name == "themes_apply_btn":
            assert "BrandPrimaryButton" in class_tag
        elif name == "themes_reset_btn":
            assert "BrandDangerButton" in class_tag
        elif name == "themes_revert_btn":
            assert "BrandCautionButton" in class_tag
            assert "PrimaryActionButton" not in class_tag
        else:
            assert "BrandSecondaryButton" in class_tag
            assert "PrimaryActionButton" not in class_tag

    settings._ensure_themes_preview_initialized()
    theme = settings.window().theme_manager.current
    reset = settings.themes_reset_btn
    reset_color = _qss_label_color(reset.styleSheet())
    assert reset_color == brand_label_color(BRAND_DANGER, theme)
    revert = settings.themes_revert_btn
    label_color = _qss_label_color(revert.styleSheet())
    expected = brand_label_color(BRAND_CAUTION, theme)
    assert label_color == expected
    assert not revert.icon().isNull()
    assert _icon_has_light_pixels(revert.icon())
    assert _icon_has_light_pixels(
        revert.icon(), mode=QIcon.Mode.Disabled
    ), "disabled revert icon should match muted label color"

    settings.refresh_menu_themes(is_dark=True)
    for name in (
        "themes_reset_btn",
        "themes_revert_btn",
        "themes_cancel_btn",
        "themes_apply_btn",
    ):
        btn = getattr(settings, name)
        assert len(btn.styleSheet()) > 100


def test_themes_colors_action_buttons_stay_branded(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_components_preview_initialized()

    for name, object_name in (
        ("themes_colors_reset_btn", "ThemesColorsResetButton"),
        ("themes_colors_revert_btn", "ThemesColorsRevertButton"),
        ("themes_colors_cancel_btn", "ThemesColorsCancelButton"),
        ("themes_colors_apply_btn", "ThemesColorsApplyButton"),
    ):
        btn = getattr(settings, name)
        assert btn is not None
        assert btn.objectName() == object_name
        ss = btn.styleSheet()
        assert len(ss) > 100, f"{name} missing widget-level brand QSS"
        assert f"#{object_name}" in ss, f"{name} QSS should scope to object name"


def test_themes_colors_reset_clears_draft_to_preset_defaults(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._on_themes_color_changed("accent", "#ff0000")
    assert settings.themes_colors_reset_btn.isEnabled()

    settings._on_themes_colors_reset_clicked()
    assert settings._themes_colors_draft_at_preset_defaults()
    assert not settings.themes_colors_reset_btn.isEnabled()


def test_themes_chat_reset_sets_wallpaper_draft_to_theme_default(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    assert settings.themes_reset_btn.isEnabled()

    settings._on_themes_chat_reset_clicked()
    assert settings._draft_surface_profile_at_default(SURFACE_CHAT_TRANSCRIPT)
    assert not settings.themes_reset_btn.isEnabled()


def test_themes_library_reset_sets_wallpaper_draft_to_theme_default(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._set_draft_surface_profile(
        SURFACE_LIBRARY_PREVIEW,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    assert settings.themes_library_reset_btn.isEnabled()

    settings._on_themes_library_reset_clicked()
    assert settings._draft_surface_profile_at_default(SURFACE_LIBRARY_PREVIEW)
    assert not settings.themes_library_reset_btn.isEnabled()


def test_themes_colors_apply_disabled_until_color_draft_changes(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    assert not settings.themes_colors_apply_btn.isEnabled()
    assert not settings.themes_colors_revert_btn.isEnabled()
    assert not settings.themes_colors_cancel_btn.isEnabled()
    assert not settings.themes_apply_btn.isEnabled()

    settings._themes_draft_overrides = {"accent": "#ff0000"}
    settings._update_themes_action_buttons()
    assert settings.themes_colors_apply_btn.isEnabled()
    assert settings.themes_colors_revert_btn.isEnabled()
    assert settings.themes_colors_cancel_btn.isEnabled()
    assert not settings.themes_apply_btn.isEnabled()

    settings._sync_colors_draft_from_applied()
    assert not settings.themes_colors_apply_btn.isEnabled()
    assert not settings.themes_colors_revert_btn.isEnabled()
    assert not settings.themes_colors_cancel_btn.isEnabled()


def test_themes_colors_apply_reenables_after_revert_to_original_color(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    base_accent = settings._base_core_values()["accent"]
    settings._on_themes_color_changed("accent", "#ff0000")
    assert settings.themes_colors_apply_btn.isEnabled()
    assert not settings.themes_apply_btn.isEnabled()

    settings._on_themes_color_changed("accent", base_accent)
    assert not settings.themes_colors_apply_btn.isEnabled()


def test_themes_apply_tracks_wallpaper_draft_and_revert(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    assert settings.themes_apply_btn.isEnabled()
    assert not settings.themes_colors_apply_btn.isEnabled()

    settings._on_themes_revert_clicked()
    assert not settings.themes_apply_btn.isEnabled()


def test_themes_chat_revert_preserves_colors_draft(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._themes_draft_overrides = {"accent": "#ff0000"}
    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    settings._update_themes_action_buttons()

    settings._on_themes_revert_clicked()

    assert settings._themes_colors_draft_is_dirty()
    assert not settings._surface_profile_dirty(SURFACE_CHAT_TRANSCRIPT)


def test_themes_colors_revert_preserves_chat_wallpaper_draft(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._themes_draft_overrides = {"accent": "#ff0000"}
    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    settings._update_themes_action_buttons()

    settings._on_themes_colors_revert_clicked()

    assert settings._surface_profile_dirty(SURFACE_CHAT_TRANSCRIPT)
    assert not settings._themes_colors_draft_is_dirty()


def test_themes_library_action_buttons_stay_branded(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()

    for name, object_name in (
        ("themes_library_reset_btn", "ThemesLibraryResetButton"),
        ("themes_library_revert_btn", "ThemesLibraryRevertButton"),
        ("themes_library_cancel_btn", "ThemesLibraryCancelButton"),
        ("themes_library_apply_btn", "ThemesLibraryApplyButton"),
    ):
        btn = getattr(settings, name)
        assert btn is not None
        assert btn.objectName() == object_name
        ss = btn.styleSheet()
        assert len(ss) > 100, f"{name} missing widget-level brand QSS"
        assert f"#{object_name}" in ss, f"{name} QSS should scope to object name"


def test_themes_library_apply_disabled_until_wallpaper_draft_changes(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    assert not settings.themes_library_apply_btn.isEnabled()
    assert not settings.themes_library_revert_btn.isEnabled()
    assert not settings.themes_library_cancel_btn.isEnabled()

    settings._set_draft_surface_profile(
        SURFACE_LIBRARY_PREVIEW,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    assert settings.themes_library_apply_btn.isEnabled()
    assert settings.themes_library_revert_btn.isEnabled()
    assert settings.themes_library_cancel_btn.isEnabled()

    settings._on_themes_library_revert_clicked()
    assert not settings.themes_library_apply_btn.isEnabled()


def test_themes_chat_revert_preserves_library_wallpaper_draft(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._set_draft_surface_profile(
        SURFACE_LIBRARY_PREVIEW,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )

    settings._on_themes_revert_clicked()

    assert settings._surface_profile_dirty(SURFACE_LIBRARY_PREVIEW)
    assert not settings._surface_profile_dirty(SURFACE_CHAT_TRANSCRIPT)


def test_themes_reading_font_apply_persists_and_enables_buttons(main_window):
    from core.app_settings import get_ui_reading_font
    from core.reading_fonts import READING_FONT_LITERATA

    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    assert not settings.themes_reading_font_apply_btn.isEnabled()

    settings._on_themes_reading_font_selected(READING_FONT_LITERATA)
    assert settings.themes_reading_font_apply_btn.isEnabled()

    settings._on_themes_reading_font_apply_clicked()
    assert get_ui_reading_font() == READING_FONT_LITERATA
    assert not settings.themes_reading_font_apply_btn.isEnabled()

    settings._on_themes_reading_font_selected(READING_FONT_LITERATA)
    settings._on_themes_reading_font_revert_clicked()
    assert not settings.themes_reading_font_apply_btn.isEnabled()


def test_themes_reading_font_browse_keeps_selected_family_label(main_window, monkeypatch):
    from core.reading_fonts import (
        READING_FONT_BROWSE_SYSTEM,
        READING_FONT_BROWSE_SYSTEM_LABEL,
        reset_reading_font_cache_for_tests,
    )

    reset_reading_font_cache_for_tests()
    monkeypatch.setattr(
        "core.reading_fonts.QFontDatabase.families",
        lambda: ["Courier New"],
    )

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return self.DialogCode.Accepted

        def selected_family(self):
            return "Courier New"

    monkeypatch.setattr(
        "ui.components.reading_font_picker_dialog.ReadingFontPickerDialog",
        _FakeDialog,
    )

    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()
    selector = settings.themes_reading_font_selector

    settings._handle_selection(
        selector,
        READING_FONT_BROWSE_SYSTEM_LABEL,
        READING_FONT_BROWSE_SYSTEM,
        settings._on_themes_reading_font_selected,
    )

    assert selector.text() == "Courier New (system)"
    assert settings._draft_reading_font_id() == "system:Courier New"
