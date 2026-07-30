"""Smoke tests for the Library page guided tour."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import build_tour


class TestLibraryTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self):
        from PyQt6.QtWidgets import (
            QLineEdit,
            QMenu,
            QPushButton,
            QTextEdit,
            QWidget,
        )

        host = QWidget()
        lv = QWidget(host)
        host.nav_library = QPushButton(host)
        host._library_view = lv
        host.ensure_library_view = MagicMock(return_value=lv)
        host._route_view = MagicMock()
        host.begin_library_chat_fab_tutorial_preview = MagicMock()
        host.end_library_chat_fab_tutorial_preview = MagicMock()

        lv.new_folder_btn = QPushButton(lv)
        lv.sort_btn = QPushButton(lv)
        lv.sort_btn.setMenu(QMenu(lv))
        lv.add_btn = QPushButton(lv)
        lv.search_bar = QLineEdit(lv)
        lv.doc_list = QPushButton(lv)
        lv.font_minus_btn = QPushButton(lv)
        lv.font_plus_btn = QPushButton(lv)
        lv.line_height_btn = QPushButton(lv)
        lv.text_align_btn = QPushButton(lv)
        lv.reader_focus_btn = QPushButton(lv)
        lv.high_contrast_btn = QPushButton(lv)
        lv.layout_mode_btn = QPushButton(lv)
        lv._preview_header_width_host = QPushButton(lv)
        lv.text_preview = QTextEdit(lv)
        lv._chat_with_doc_btn = QPushButton(lv)

        return host

    def test_step_count_matches_plan_blocks(self) -> None:
        tour = build_tour("library", self._make_host())
        self.assertIsNotNone(tour)
        assert tour is not None
        # welcome + 5 sidebar + 1 indexing mode + 7 toolbar + 2 preview + 1 fab + 1 finish = 18
        self.assertEqual(len(tour._steps), 18)

    def test_step_order_section_anchors(self) -> None:
        tour = build_tour("library", self._make_host())
        assert tour is not None
        ids = [step.step_id for step in tour._steps]
        self.assertEqual(ids[0], "welcome")
        self.assertEqual(
            ids[1:7],
            [
                "sidebar_new_folder",
                "sidebar_sort",
                "sidebar_ingest",
                "sidebar_indexing_mode",
                "sidebar_search",
                "sidebar_doc_list",
            ],
        )
        self.assertEqual(ids[7], "preview_font_minus")
        self.assertEqual(ids[14], "preview_header")
        self.assertEqual(ids[15], "preview_body")
        self.assertEqual(ids[-2], "chat_with_doc")
        self.assertEqual(ids[-1], "tour_complete")

    def test_all_target_getters_resolve(self) -> None:
        host = self._make_host()
        tour = build_tour("library", host)
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

    def test_on_enter_routes_to_library(self) -> None:
        host = self._make_host()
        tour = build_tour("library", host)
        assert tour is not None
        tour._steps[0].on_enter(host)
        host._route_view.assert_called_with(1, host.nav_library)

    def test_chat_fab_step_calls_begin_preview(self) -> None:
        host = self._make_host()
        tour = build_tour("library", host)
        assert tour is not None
        fab_step = next(s for s in tour._steps if s.step_id == "chat_with_doc")
        fab_step.on_enter(host)
        host.begin_library_chat_fab_tutorial_preview.assert_called_once()

    def test_on_finished_dismisses_transients(self) -> None:
        host = self._make_host()
        tour = build_tour("library", host)
        assert tour is not None
        tour.start()
        tour.finish()
        host.end_library_chat_fab_tutorial_preview.assert_called()


if __name__ == "__main__":
    unittest.main()
