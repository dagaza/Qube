"""Themes Revert/Cancel/Apply brand button styling."""

from __future__ import annotations


def test_themes_action_buttons_stay_branded_and_enabled(main_window):
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
        assert btn.isEnabled()
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
        assert btn.isEnabled()
        assert len(btn.styleSheet()) > 100

    settings._themes_draft_overrides = {"accent": "#ff0000"}
    settings._update_themes_action_buttons()
    assert settings.themes_apply_btn.isEnabled()
    assert len(settings.themes_apply_btn.styleSheet()) > 100
