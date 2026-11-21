"""Tray / action menu theming must include QMenu::item rules (Linux system tray)."""

from core.theme.accessors import theme_for
from core.theme.widget_styles import SETTINGS_PRESTIGE_MENU
from ui.shell_theme import prestige_action_menu_qss


def test_action_menu_qss_includes_item_rules():
    theme = theme_for(is_dark=False)
    qss = prestige_action_menu_qss(theme)
    assert "QMenu::item" in qss
    assert theme.text_primary in qss
    assert "PrestigeMenuList" not in qss


def test_prestige_list_menu_qss_targets_embedded_list():
    theme = theme_for(is_dark=True)
    qss = theme.style(SETTINGS_PRESTIGE_MENU)
    assert "PrestigeMenuList" in qss
    assert "QMenu::item" not in qss