"""Type-to-search keyboard routing helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QPushButton, QWidget

from ui.components.type_to_search import (
    focus_target_with_key,
    is_type_to_search_key,
    should_handle_type_to_focus,
)


def _key(text: str = "", *, key: Qt.Key = Qt.Key.Key_unknown, modifiers=Qt.KeyboardModifier.NoModifier) -> QKeyEvent:
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, modifiers, text)


class TestTypeToSearchKeys(unittest.TestCase):
    def test_printable_character_is_candidate(self) -> None:
        self.assertTrue(is_type_to_search_key(_key("a")))

    def test_space_is_candidate(self) -> None:
        self.assertTrue(is_type_to_search_key(_key(" ")))

    def test_ctrl_combo_is_ignored(self) -> None:
        self.assertFalse(
            is_type_to_search_key(
                _key("c", modifiers=Qt.KeyboardModifier.ControlModifier)
            )
        )

    def test_navigation_keys_are_ignored(self) -> None:
        self.assertFalse(is_type_to_search_key(_key(key=Qt.Key.Key_Down)))
        self.assertFalse(is_type_to_search_key(_key(key=Qt.Key.Key_Escape)))

    def test_backspace_is_ignored(self) -> None:
        self.assertFalse(is_type_to_search_key(_key(key=Qt.Key.Key_Backspace)))


class TestShouldHandleTypeToSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_redirects_when_host_visible_and_focus_on_non_text_widget(self) -> None:
        host = QWidget()
        search = QLineEdit(host)
        button = QPushButton("Filter", host)
        host.show()
        button.setFocus()
        event = _key("m")
        self.assertTrue(should_handle_type_to_focus(host, search, event))

    def test_skips_when_search_already_focused(self) -> None:
        host = QWidget()
        search = QLineEdit(host)
        host.show()
        search.setFocus()
        event = _key("m")
        self.assertFalse(should_handle_type_to_focus(host, search, event))

    def test_skips_when_other_text_field_focused(self) -> None:
        host = QWidget()
        search = QLineEdit(host)
        other = QLineEdit(host)
        host.show()
        other.setFocus()
        event = _key("m")
        self.assertFalse(should_handle_type_to_focus(host, search, event))

    def test_skips_when_modal_open(self) -> None:
        host = QWidget()
        search = QLineEdit(host)
        host.show()
        event = _key("s")
        modal = MagicMock()
        with patch.object(QApplication, "activeModalWidget", return_value=modal):
            self.assertFalse(should_handle_type_to_focus(host, search, event))

    def test_skips_when_onboarding_tour_active(self) -> None:
        host = QWidget()
        search = QLineEdit(host)
        host.show()
        tour = MagicMock()
        tour.is_active = True
        host._active_tour = tour  # type: ignore[attr-defined]
        event = _key("s")
        self.assertFalse(should_handle_type_to_focus(host, search, event))

    def test_redirects_to_composer_when_focus_on_button(self) -> None:
        host = QWidget()
        composer = QPlainTextEdit(host)
        button = QPushButton("Send", host)
        host.show()
        button.setFocus()
        event = _key("h")
        self.assertTrue(should_handle_type_to_focus(host, composer, event))

    def test_skips_when_history_search_focused(self) -> None:
        host = QWidget()
        composer = QPlainTextEdit(host)
        history_search = QLineEdit(host)
        host.show()
        history_search.setFocus()
        event = _key("h")
        self.assertFalse(should_handle_type_to_focus(host, composer, event))

    def test_inserts_into_plain_text_composer(self) -> None:
        composer = QPlainTextEdit()
        composer.show()
        focus_target_with_key(composer, _key("a"))
        self.assertEqual(composer.toPlainText(), "a")
