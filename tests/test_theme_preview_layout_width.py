"""Measure theme preview layout widths at minimum window size."""

from __future__ import annotations

from ui.components.theme_preview_panel import (
    _PREVIEW_LAYOUT_MIN_WIDTH,
    _design_preview_width_at_min_window,
    _preview_card_inner_width,
    _preview_scroll_viewport_width,
)


def _wait_for_preview_card_layout(qtbot, panel) -> int:
    """Wait until the settings card has a stable inner width for assertions."""

    def ready() -> bool:
        card_inner = _preview_card_inner_width(panel)
        return (
            card_inner is not None
            and card_inner >= _PREVIEW_LAYOUT_MIN_WIDTH
            and panel.width() > 0
        )

    qtbot.waitUntil(ready, timeout=5000)
    card_inner = _preview_card_inner_width(panel)
    assert card_inner is not None
    return card_inner


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
    viewport_w = _preview_scroll_viewport_width(panel) or 0

    design = _design_preview_width_at_min_window()
    card_inner = _wait_for_preview_card_layout(qtbot, panel)
    pixmap = panel._conversations_view.grab()

    assert pixmap.width() == panel.width()
    assert panel.width() <= design + 2
    assert pixmap.width() <= card_inner + 2, (
        f"pixmap={pixmap.width()} card_inner={card_inner} panel={panel.width()} "
        f"viewport={viewport_w} design={design}"
    )
    if viewport_w >= _PREVIEW_LAYOUT_MIN_WIDTH:
        assert pixmap.width() <= viewport_w + 2, (
            f"pixmap={pixmap.width()} viewport={viewport_w} card_inner={card_inner}"
        )


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
    viewport_w = _preview_scroll_viewport_width(panel) or 0

    design = _design_preview_width_at_min_window()
    card_inner = _wait_for_preview_card_layout(qtbot, panel)
    pixmap = panel._view.grab()

    assert pixmap.width() == panel.width()
    assert panel.width() <= design + 2
    assert pixmap.width() <= card_inner + 2
    if viewport_w >= _PREVIEW_LAYOUT_MIN_WIDTH:
        assert pixmap.width() <= viewport_w + 2


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
