"""Themes Revert/Cancel/Apply brand button styling and draft-aware enabled state."""

from __future__ import annotations

from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
from core.surface_fill.models import SurfaceProfile, WallpaperNone


def test_themes_action_buttons_stay_branded(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()

    for name, object_name in (
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
        elif name == "themes_revert_btn":
            assert "BrandCautionButton" in class_tag
            assert "PrimaryActionButton" not in class_tag
        else:
            assert "BrandSecondaryButton" in class_tag
            assert "PrimaryActionButton" not in class_tag

    settings.refresh_menu_themes(is_dark=True)
    for name in ("themes_revert_btn", "themes_cancel_btn", "themes_apply_btn"):
        btn = getattr(settings, name)
        assert len(btn.styleSheet()) > 100


def test_themes_apply_disabled_until_draft_changes(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    assert not settings.themes_apply_btn.isEnabled()
    assert not settings.themes_revert_btn.isEnabled()
    assert not settings.themes_cancel_btn.isEnabled()

    settings._themes_draft_overrides = {"accent": "#ff0000"}
    settings._update_themes_action_buttons()
    assert settings.themes_apply_btn.isEnabled()
    assert settings.themes_revert_btn.isEnabled()
    assert settings.themes_cancel_btn.isEnabled()

    settings._sync_themes_draft_from_applied()
    assert not settings.themes_apply_btn.isEnabled()
    assert not settings.themes_revert_btn.isEnabled()
    assert not settings.themes_cancel_btn.isEnabled()


def test_themes_apply_reenables_after_revert_to_original_color(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    base_accent = settings._base_core_values()["accent"]
    settings._on_themes_color_changed("accent", "#ff0000")
    assert settings.themes_apply_btn.isEnabled()

    settings._on_themes_color_changed("accent", base_accent)
    assert not settings.themes_apply_btn.isEnabled()


def test_themes_apply_tracks_wallpaper_draft_and_revert(main_window):
    settings = main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )
    assert settings.themes_apply_btn.isEnabled()

    settings._on_themes_revert_clicked()
    assert not settings.themes_apply_btn.isEnabled()
