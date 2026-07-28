"""Page guided tour registry and helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import (
    build_tour,
    get_tour_builder,
    list_registered_tour_ids,
    settings_section_tour_id,
    tour_display_name,
)


class TestPageTourRegistry(unittest.TestCase):
    def test_all_page_tours_registered(self) -> None:
        ids = list_registered_tour_ids()
        expected = {
            "conversations",
            "library",
            "memory_manager",
            "model_manager",
            "telemetry",
            "settings.voice_audio",
            "settings.ai_models",
            "settings.memory",
            "settings.knowledge",
            "settings.general",
            "settings.appearance_themes",
            "settings.companion_desktop",
            "settings.notifications",
            "settings.help",
            "settings.contact_feedback",
            "settings.advanced",
        }
        self.assertEqual(set(ids), expected)

    def test_settings_section_tour_id_mapping(self) -> None:
        self.assertEqual(settings_section_tour_id("voice.audio"), "settings.voice_audio")
        self.assertEqual(settings_section_tour_id("ai.models"), "settings.ai_models")
        self.assertEqual(
            settings_section_tour_id("companion.desktop"),
            "settings.companion_desktop",
        )

    def test_tour_display_name(self) -> None:
        self.assertEqual(tour_display_name("conversations"), "Conversations")
        self.assertIn("Telemetry", tour_display_name("telemetry"))

    def test_build_tour_unknown_returns_none(self) -> None:
        host = MagicMock()
        self.assertIsNone(build_tour("missing.tour", host))

    def test_build_conversations_tour_has_steps(self) -> None:
        from PyQt6.QtWidgets import QPushButton, QWidget

        host = QWidget()
        host.nav_chat = QPushButton(host)
        cv = QWidget(host)
        host.conversations_view = cv
        cv.new_chat_btn = QPushButton(cv)
        cv.search_bar = QPushButton(cv)
        cv.history_list = QPushButton(cv)
        cv.sort_btn = QPushButton(cv)
        cv.font_minus_btn = QPushButton(cv)
        cv.conversation_download_btn = QPushButton(cv)
        cv.text_input = QPushButton(cv)
        cv.web_btn = QPushButton(cv)
        cv.composer_attach_btn = QPushButton(cv)
        cv.send_btn = QPushButton(cv)
        host.mic_selector_btn = QPushButton(host)
        host.status_bubble = QPushButton(host)
        host.rag_status_dot = QPushButton(host)
        host.toolbar_native_model_selector = QPushButton(host)
        host.voice_input_toggle = QPushButton(host)
        host.toolbar_timeout_spin = QPushButton(host)
        host._route_view = MagicMock()
        host.tools_content = MagicMock()
        host.tools_content.maximumWidth.return_value = 260
        host._toggle_tools_pane = MagicMock()

        tour = build_tour("conversations", host)
        self.assertIsNotNone(tour)
        assert tour is not None
        self.assertGreaterEqual(len(tour._steps), 10)


class TestPageTourRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_build_page_tour_returns_none_without_host(self) -> None:
        from ui.onboarding.tour_runner import build_page_tour

        parent = MagicMock()
        parent.window.return_value = None
        self.assertIsNone(build_page_tour(parent, "conversations"))

    def test_build_page_tour_builds_registered_tour(self) -> None:
        from PyQt6.QtWidgets import QPushButton, QWidget

        from ui.onboarding.tour_runner import build_page_tour

        host = QWidget()
        host.nav_library = QPushButton(host)
        lv = QWidget(host)
        host._library_view = lv
        host.ensure_library_view = MagicMock(return_value=lv)
        lv.library_list = MagicMock()
        lv.preview_toolbar = QPushButton(lv)
        lv.metadata_panel = QPushButton(lv)
        lv.preview_panel = QPushButton(lv)
        host._route_view = MagicMock()

        tour = build_page_tour(host, "library")
        self.assertIsNotNone(tour)


class TestMainWindowTourManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_finish_active_tour_clears_active(self) -> None:
        from ui.main_window import MainWindow

        host = MagicMock()
        mock_tour = MagicMock()
        mock_tour.is_active = True
        host._active_tour = mock_tour
        MainWindow.finish_active_tour(host)
        mock_tour.finish.assert_called_once()
        self.assertIsNone(host._active_tour)


if __name__ == "__main__":
    unittest.main()
