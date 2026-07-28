"""Tests for collapsible Settings section cards."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from core.app_settings import (
    KEY_UI_SETTINGS_SECTION_CARDS_COLLAPSIBLE,
    KEY_UI_SETTINGS_SECTION_CARDS_DEFAULT_EXPANDED,
    get_settings_section_cards_collapsible,
    get_settings_section_cards_default_expanded,
    set_settings_section_cards_collapsible,
    set_settings_section_cards_default_expanded,
)
from ui.views.settings.settings_card_style import (
    begin_settings_section_card,
    set_settings_collapsible_cards_expanded,
    sync_settings_collapsible_cards,
)
from ui.views.settings.widgets import add_subsection_to_layout


class TestSettingsCollapsibleCards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        set_settings_section_cards_collapsible(True)
        set_settings_section_cards_default_expanded(True)

    def test_schema_keys_have_getters(self) -> None:
        self.assertTrue(get_settings_section_cards_collapsible())
        self.assertTrue(get_settings_section_cards_default_expanded())
        self.assertEqual(
            KEY_UI_SETTINGS_SECTION_CARDS_COLLAPSIBLE,
            "qube.ui.settings_section_cards_collapsible",
        )
        self.assertEqual(
            KEY_UI_SETTINGS_SECTION_CARDS_DEFAULT_EXPANDED,
            "qube.ui.settings_section_cards_default_expanded",
        )

    def test_subsection_title_moves_to_card_header(self) -> None:
        host = QWidget()
        host._current_settings_section_id = "knowledge"
        host._settings_section_cards = []
        host._settings_collapsible_cards_by_section = {}

        page = QWidget()
        page_layout = QVBoxLayout(page)
        wrapper, card_layout = begin_settings_section_card(host, is_dark=True)
        add_subsection_to_layout(card_layout, "Search quality", anchor="embedding_mode")
        page_layout.addWidget(wrapper)

        handles = host._settings_collapsible_cards_by_section["knowledge"]
        self.assertEqual(len(handles), 1)
        handle = handles[0]
        self.assertEqual(handle.title_lbl.text(), "Search quality")
        self.assertEqual(handle.title_lbl.property("settings_anchor"), "embedding_mode")
        self.assertTrue(handle.expanded)

    def test_default_collapsed_when_setting_false(self) -> None:
        set_settings_section_cards_default_expanded(False)
        host = QWidget()
        host._current_settings_section_id = "memory"
        host._settings_section_cards = []
        host._settings_collapsible_cards_by_section = {}

        wrapper, card_layout = begin_settings_section_card(host, is_dark=True)
        add_subsection_to_layout(card_layout, "Pipeline")
        handle = host._settings_collapsible_cards_by_section["memory"][0]
        self.assertFalse(handle.expanded)
        self.assertFalse(handle.card.isVisible())

    def test_page_level_expand_collapse_all(self) -> None:
        host = QWidget()
        host._current_settings_section_id = "help"
        host._settings_section_cards = []
        host._settings_collapsible_cards_by_section = {}
        host._sync_settings_collapse_all_button = MagicMock()

        page = QWidget()
        page_layout = QVBoxLayout(page)
        for title in ("Docs", "Tours"):
            wrapper, card_layout = begin_settings_section_card(host, is_dark=True)
            add_subsection_to_layout(card_layout, title)
            page_layout.addWidget(wrapper)

        set_settings_collapsible_cards_expanded(host, "help", expanded=False)
        handles = host._settings_collapsible_cards_by_section["help"]
        self.assertFalse(any(h.expanded for h in handles))

        set_settings_collapsible_cards_expanded(host, "help", expanded=True)
        self.assertTrue(all(h.expanded for h in handles))

    def test_collapsible_disabled_hides_headers(self) -> None:
        host = QWidget()
        host._current_settings_section_id = "general"
        host._settings_section_cards = []
        host._settings_collapsible_cards_by_section = {}
        host._sync_settings_collapse_all_button = MagicMock()

        wrapper, card_layout = begin_settings_section_card(host, is_dark=True)
        add_subsection_to_layout(card_layout, "Language")
        page = QWidget()
        QVBoxLayout(page).addWidget(wrapper)
        page.show()
        self._app.processEvents()
        handle = host._settings_collapsible_cards_by_section["general"][0]
        handle.set_expanded(False)

        set_settings_section_cards_collapsible(False)
        sync_settings_collapsible_cards(host, is_dark=True)
        self.assertFalse(handle.header.isVisible())
        self.assertTrue(handle.card.isVisible())


if __name__ == "__main__":
    unittest.main()
