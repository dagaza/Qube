"""Measure theme preview layout widths at minimum window size."""

from __future__ import annotations

from ui.views.settings.settings_card_style import settings_card_content_horizontal_padding_total
from ui.components.theme_preview_panel import _design_preview_width_at_min_window


def _open_themes_preview(main_window, qtbot):
    win = main_window
    win._set_tools_pane_expanded(False, animate=False)
    settings = win.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()
    qtbot.wait(200)
    settings._refresh_themes_preview()
    qtbot.wait(200)
    return settings


def _open_themes_library_preview(main_window, qtbot):
    win = main_window
    win._set_tools_pane_expanded(False, animate=False)
    settings = win.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_library_preview_initialized()
    qtbot.wait(200)
    settings._refresh_themes_library_preview()
    qtbot.wait(200)
    return settings


def test_theme_preview_fits_settings_card_at_min_window(main_window, qtbot):
    win = main_window
    win.resize(1200, 950)
    qtbot.wait(50)
    settings = _open_themes_preview(win, qtbot)

    panel = settings.themes_preview_panel
    card = settings.themes_preview_card
    scroll = settings.settings_section_stack.currentWidget()
    viewport_w = scroll.viewport().width() if scroll is not None else 0
    pixmap = panel._conversations_view.grab()

    design = _design_preview_width_at_min_window()
    card_inner = card.width() - settings_card_content_horizontal_padding_total()
    assert pixmap.width() <= card_inner + 2, (
        f"pixmap={pixmap.width()} card_inner={card_inner} panel={panel.width()} "
        f"viewport={viewport_w} design={design}"
    )
    assert pixmap.width() <= viewport_w + 2, (
        f"pixmap={pixmap.width()} viewport={viewport_w} card_inner={card_inner}"
    )
    assert pixmap.width() == design


def test_theme_preview_stays_capped_on_wide_window(main_window, qtbot):
    win = main_window
    win.resize(1600, 950)
    qtbot.wait(50)
    settings = _open_themes_preview(win, qtbot)

    panel = settings.themes_preview_panel
    pixmap = panel._conversations_view.grab()
    design = _design_preview_width_at_min_window()

    assert pixmap.width() == design
    assert panel.width() == design


def test_library_preview_fits_settings_card_at_min_window(main_window, qtbot):
    win = main_window
    win.resize(1200, 950)
    qtbot.wait(50)
    settings = _open_themes_library_preview(win, qtbot)

    panel = settings.themes_library_preview_panel
    card = settings.themes_library_preview_card
    scroll = settings.settings_section_stack.currentWidget()
    viewport_w = scroll.viewport().width() if scroll is not None else 0
    pixmap = panel._view.grab()

    design = _design_preview_width_at_min_window()
    card_inner = card.width() - settings_card_content_horizontal_padding_total()
    assert pixmap.width() <= card_inner + 2
    assert pixmap.width() <= viewport_w + 2
    assert pixmap.width() == design


def test_library_preview_stays_capped_on_wide_window(main_window, qtbot):
    win = main_window
    win.resize(1600, 950)
    qtbot.wait(50)
    settings = _open_themes_library_preview(win, qtbot)

    panel = settings.themes_library_preview_panel
    pixmap = panel._view.grab()
    design = _design_preview_width_at_min_window()

    assert pixmap.width() == design
    assert panel.width() == design
