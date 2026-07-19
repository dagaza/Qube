"""Smoke tests for the Memory Manager page guided tour."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import build_tour


class TestMemoryManagerTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self):
        from PyQt6.QtWidgets import QFrame, QLineEdit, QMenu, QPushButton, QWidget

        host = QWidget()
        mv = QWidget(host)
        host.nav_memory = QPushButton(host)
        host._memory_manager_view = mv
        host.ensure_memory_manager_view = MagicMock(return_value=mv)
        host._route_view = MagicMock()
        host.begin_memory_themes_tutorial_preview = MagicMock()
        host.end_memory_themes_tutorial_preview = MagicMock()

        mv.profile_card = QFrame(mv)
        mv.tier_selector = QPushButton(mv)
        mv.tier_selector.setMenu(QMenu(mv))
        mv.category_selector = QPushButton(mv)
        mv.category_selector.setMenu(QMenu(mv))
        mv.flagged_btn = QPushButton(mv)
        mv.search_input = QLineEdit(mv)
        mv.bulk_delete_btn = QPushButton(mv)
        mv.export_btn = QPushButton(mv)
        mv.themes_card = QFrame(mv)
        mv.scroll = QPushButton(mv)
        mv.refresh_btn = QPushButton(mv)

        return host

    def test_step_count_matches_plan_blocks(self) -> None:
        tour = build_tour("memory_manager", self._make_host())
        self.assertIsNotNone(tour)
        assert tour is not None
        # welcome + 10 mainstage + 1 finish = 12
        self.assertEqual(len(tour._steps), 12)

    def test_step_order_section_anchors(self) -> None:
        tour = build_tour("memory_manager", self._make_host())
        assert tour is not None
        ids = [step.step_id for step in tour._steps]
        self.assertEqual(ids[0], "welcome")
        self.assertEqual(ids[1], "profile")
        self.assertEqual(ids[2:4], ["tier_filter", "category_filter"])
        self.assertEqual(ids[-2], "refresh")
        self.assertEqual(ids[-1], "tour_complete")

    def test_all_target_getters_resolve(self) -> None:
        host = self._make_host()
        tour = build_tour("memory_manager", host)
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

    def test_on_enter_routes_to_memory_manager(self) -> None:
        host = self._make_host()
        tour = build_tour("memory_manager", host)
        assert tour is not None
        tour._steps[0].on_enter(host)
        host._route_view.assert_called_with(2, host.nav_memory)

    def test_themes_step_calls_begin_preview(self) -> None:
        host = self._make_host()
        tour = build_tour("memory_manager", host)
        assert tour is not None
        themes_step = next(s for s in tour._steps if s.step_id == "themes")
        themes_step.on_enter(host)
        host.begin_memory_themes_tutorial_preview.assert_called_once()

    def test_on_finished_dismisses_transients(self) -> None:
        host = self._make_host()
        tour = build_tour("memory_manager", host)
        assert tour is not None
        tour.start()
        tour.finish()
        host.end_memory_themes_tutorial_preview.assert_called()


if __name__ == "__main__":
    unittest.main()
