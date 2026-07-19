"""Smoke tests for the Model Manager page guided tour."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import build_tour


class TestModelManagerTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self):
        from PyQt6.QtWidgets import (
            QFrame,
            QLineEdit,
            QListWidget,
            QPushButton,
            QTextBrowser,
            QWidget,
        )

        host = QWidget()
        mm = QWidget(host)
        host.nav_models = QPushButton(host)
        host._model_manager_view = mm
        host.ensure_model_manager_view = MagicMock(return_value=mm)
        host._route_view = MagicMock()
        mm.begin_load_more_tutorial_preview = MagicMock()
        mm.end_load_more_tutorial_preview = MagicMock()

        mm.hub_search_edit = QLineEdit(mm)
        mm.hub_list_hint = QPushButton(mm)
        mm.hub_model_list = QListWidget(mm)
        mm.hub_load_more_btn = QPushButton(mm)
        mm.detail_title = QPushButton(mm)
        mm.detail_source_btn = QPushButton(mm)
        mm.meta_panel = QFrame(mm)
        mm.hf_file_combo = QPushButton(mm)
        mm.system_chip_lbl = QPushButton(mm)
        mm.download_btn = QPushButton(mm)
        mm.readme_browser = QTextBrowser(mm)

        return host

    def test_step_count_matches_plan_blocks(self) -> None:
        tour = build_tour("model_manager", self._make_host())
        self.assertIsNotNone(tour)
        assert tour is not None
        # welcome + 4 hub + 7 detail + 1 finish = 13
        self.assertEqual(len(tour._steps), 13)

    def test_step_order_section_anchors(self) -> None:
        tour = build_tour("model_manager", self._make_host())
        assert tour is not None
        ids = [step.step_id for step in tour._steps]
        self.assertEqual(ids[0], "welcome")
        self.assertEqual(
            ids[1:5],
            ["hub_search", "hub_list_hint", "hub_list", "hub_load_more"],
        )
        self.assertEqual(ids[5], "detail_title")
        self.assertEqual(ids[-2], "detail_readme")
        self.assertEqual(ids[-1], "tour_complete")

    def test_all_target_getters_resolve(self) -> None:
        host = self._make_host()
        tour = build_tour("model_manager", host)
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

    def test_on_enter_routes_to_model_manager(self) -> None:
        host = self._make_host()
        tour = build_tour("model_manager", host)
        assert tour is not None
        tour._steps[0].on_enter(host)
        host._route_view.assert_called_with(4, host.nav_models)

    def test_load_more_step_calls_begin_preview(self) -> None:
        host = self._make_host()
        mm = host._model_manager_view
        tour = build_tour("model_manager", host)
        assert tour is not None
        load_more_step = next(s for s in tour._steps if s.step_id == "hub_load_more")
        load_more_step.on_enter(host)
        mm.begin_load_more_tutorial_preview.assert_called_once()

    def test_on_finished_dismisses_transients(self) -> None:
        host = self._make_host()
        mm = host._model_manager_view
        tour = build_tour("model_manager", host)
        assert tour is not None
        tour.start()
        tour.finish()
        mm.end_load_more_tutorial_preview.assert_called()


if __name__ == "__main__":
    unittest.main()
