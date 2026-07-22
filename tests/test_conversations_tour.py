"""Smoke tests for the Conversations page guided tour."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import build_tour


class TestConversationsTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self):
        from PyQt6.QtWidgets import QLineEdit, QMenu, QProgressBar, QPushButton, QWidget

        host = QWidget()
        cv = QWidget(host)
        host.nav_chat = QPushButton(host)
        host.conversations_view = cv
        host._route_view = MagicMock()
        host.tools_content = MagicMock()
        host.tools_content.maximumWidth.return_value = 260
        host._toggle_tools_pane = MagicMock()
        host.begin_ddg_backoff_tutorial_preview = MagicMock()
        host.end_ddg_backoff_tutorial_preview = MagicMock()

        cv.new_folder_btn = QPushButton(cv)
        cv.sort_btn = QPushButton(cv)
        cv.sort_btn.setMenu(QMenu(cv))
        cv.new_chat_btn = QPushButton(cv)
        cv.font_minus_btn = QPushButton(cv)
        cv.font_plus_btn = QPushButton(cv)
        cv.line_height_btn = QPushButton(cv)
        cv.text_align_btn = QPushButton(cv)
        cv.reader_focus_btn = QPushButton(cv)
        cv.high_contrast_btn = QPushButton(cv)
        cv.layout_mode_btn = QPushButton(cv)
        cv.conversation_download_btn = QPushButton(cv)
        cv.conversation_copy_btn = QPushButton(cv)
        cv.web_btn = QPushButton(cv)
        cv.think_btn = QPushButton(cv)
        cv.composer_attach_btn = QPushButton(cv)
        cv.composer_voice_btn = QPushButton(cv)
        cv.text_input = QLineEdit(cv)
        cv.send_btn = QPushButton(cv)

        host.toggle_tools_btn = QPushButton(host)
        host.toolbar_native_model_selector = QPushButton(host)
        host.toolbar_native_model_eject_btn = QPushButton(host)
        host.toolbar_auto_load_model_toggle = QPushButton(host)
        host.voice_input_toggle = QPushButton(host)
        host.toolbar_timeout_spin = QPushButton(host)
        host.toolbar_threshold_spin = QPushButton(host)
        host.toolbar_wakeword_sensitivity_spin = QPushButton(host)
        host.voice_bypass_toggle = QPushButton(host)
        host.global_voice_selector = QPushButton(host)
        host.temp_spin = QPushButton(host)
        host.ctx_spin = QPushButton(host)
        host.history_spin = QPushButton(host)
        host.max_reply_spin = QPushButton(host)
        host.tool_rag_toggle = QPushButton(host)
        host.rag_auto_toggle = QPushButton(host)
        host.rag_strict_toggle = QPushButton(host)
        host.tool_internet_hybrid_toggle = QPushButton(host)
        host.toolbar_privacy_tier_selector = QPushButton(host)

        host.topbar_mic_cluster = QWidget(host)
        host.vu_meter = QProgressBar(host.topbar_mic_cluster)
        host.status_bubble = QPushButton(host)
        host.rag_status_dot = QPushButton(host)
        host.web_status_dot = QPushButton(host)
        host.hybrid_status_dot = QPushButton(host)
        host.ddg_backoff_label = QPushButton(host)

        return host

    def test_step_count_matches_plan_blocks(self) -> None:
        tour = build_tour("conversations", self._make_host())
        self.assertIsNotNone(tour)
        assert tour is not None
        # welcome + 3 sidebar + 9 main + 6 composer + 19 tools + 5 top bar + 1 ddg + 1 finish = 45
        self.assertEqual(len(tour._steps), 45)

    def test_step_order_section_anchors(self) -> None:
        tour = build_tour("conversations", self._make_host())
        assert tour is not None
        ids = [step.step_id for step in tour._steps]
        self.assertEqual(ids[0], "welcome")
        self.assertEqual(ids[1:4], ["sidebar_new_folder", "sidebar_sort", "sidebar_new_chat"])
        self.assertEqual(ids[4], "main_font_minus")
        self.assertEqual(ids[13], "composer_web")
        self.assertEqual(ids[19], "tools_collapse")
        self.assertEqual(ids[32], "tools_max_reply_tokens")
        self.assertEqual(ids[37], "tools_privacy_tier")
        self.assertEqual(ids[38], "topbar_vu")
        self.assertEqual(ids[-2], "topbar_ddg_cooldown")
        self.assertEqual(ids[-1], "tour_complete")

    def test_all_target_getters_resolve(self) -> None:
        host = self._make_host()
        tour = build_tour("conversations", host)
        assert tour is not None
        missing: list[str] = []
        for step in tour._steps:
            if step.target_getter is None:
                continue
            target = step.target_getter(host)
            if target is None:
                missing.append(step.step_id)
            if step.on_enter is not None:
                step.on_enter(host)
        self.assertEqual(missing, [])

    def test_on_enter_routes_to_conversations(self) -> None:
        host = self._make_host()
        tour = build_tour("conversations", host)
        assert tour is not None
        tour._steps[0].on_enter(host)
        host._route_view.assert_called_with(0, host.nav_chat)

    def test_ddg_preview_step_calls_begin_preview(self) -> None:
        host = self._make_host()
        tour = build_tour("conversations", host)
        assert tour is not None
        ddg_step = next(s for s in tour._steps if s.step_id == "topbar_ddg_cooldown")
        ddg_step.on_enter(host)
        host.begin_ddg_backoff_tutorial_preview.assert_called_once()

    def test_on_finished_dismisses_transients(self) -> None:
        host = self._make_host()
        tour = build_tour("conversations", host)
        assert tour is not None
        tour.start()
        tour.finish()
        host.end_ddg_backoff_tutorial_preview.assert_called()


if __name__ == "__main__":
    unittest.main()
