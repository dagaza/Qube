"""Onboarding coach panel layout — wrapped body text must not clip."""

from __future__ import annotations

import unittest

from core.catalog_hardware_recommendation import build_tour_model_download_body
from core.hardware_capability_profile import HardwareCapabilityProfile, HardwareTier
from tests.test_catalog_hardware_recommendation import _entry
from ui.components.onboarding_tour import (
    OnboardingCoachPanel,
    _dropdown_menu_step_active,
    _resolve_panel_position,
    _scroll_target_into_view,
)


class TestOnboardingCoachPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_long_tour_body_gets_multi_line_height(self) -> None:
        profile = HardwareCapabilityProfile(
            total_ram_gb=16.0,
            total_vram_gb=4.0,
            cpu_cores=8,
            gpu_name="Test GPU",
            gpu_backend="nvidia",
            tier=HardwareTier.COMPACT,
        )
        entries = [
            _entry("gemma-4-e4b-it", "Gemma 4 4B Instruct"),
            _entry("phi-4-mini-instruct", "Phi-4 Mini Instruct"),
        ]
        body = build_tour_model_download_body(entries, profile=profile)

        panel = OnboardingCoachPanel()
        panel.body_lbl.setText(body)
        content_w = panel._content_inner_width()
        line_h = panel.body_lbl.fontMetrics().lineSpacing()
        wrapped_h = panel._label_wrapped_height(panel.body_lbl, content_w)

        self.assertGreater(wrapped_h, line_h * 3)

        panel.recalculate_content_size()
        self.assertGreaterEqual(
            panel.body_lbl.minimumHeight(),
            wrapped_h + panel._TEXT_LABEL_VERTICAL_PAD,
        )
        self.assertGreater(panel.height(), wrapped_h)

    def test_recalculate_content_size_does_not_grow_on_repeat(self) -> None:
        panel = OnboardingCoachPanel()
        panel.body_lbl.setText(
            "Pick a model that fits your GPU.\n"
            "You can change it later from Settings or Model Manager."
        )
        panel.recalculate_content_size()
        height0 = panel.height()
        min_h0 = panel.body_lbl.minimumHeight()
        for _ in range(40):
            panel.recalculate_content_size()
        self.assertEqual(panel.height(), height0)
        self.assertEqual(panel.body_lbl.minimumHeight(), min_h0)

    def test_dropdown_step_detects_open_menu(self) -> None:
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QMenu, QPushButton, QWidget

        host = QWidget()
        btn = QPushButton("Filter", host)
        menu = QMenu(btn)
        menu.addAction("Option A")
        btn.setMenu(menu)
        btn.showMenu()

        target_global = QRect(200, 300, 120, 32)
        self.assertTrue(_dropdown_menu_step_active(btn, target_global))

    def test_dropdown_step_avoids_menu_when_no_room_above(self) -> None:
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QMenu, QPushButton, QWidget

        from ui.components.onboarding_tour import _global_widget_rect

        host = QWidget()
        host.resize(900, 700)
        btn = QPushButton("Filter", host)
        btn.move(200, 24)
        btn.resize(120, 32)
        btn.show()
        menu = QMenu(btn)
        menu.addAction("Option A")
        menu.addAction("Option B")
        btn.setMenu(menu)
        btn.showMenu()
        self._app.processEvents()

        panel = OnboardingCoachPanel(host)
        panel.body_lbl.setText("Choose how folders and items are sorted.")
        panel.recalculate_content_size()

        target_global = _global_widget_rect(btn, margin=6)
        x, y = _resolve_panel_position(
            host,
            target_global,
            btn,
            panel_width=panel.width(),
            panel_height=panel.height(),
            margin=16,
        )
        panel_rect = QRect(x, y, panel.width(), panel.height())
        menu_global = _global_widget_rect(menu, margin=0)
        menu_local = QRect(
            host.mapFromGlobal(menu_global.topLeft()),
            menu_global.size(),
        )
        self.assertFalse(panel_rect.intersects(menu_local))
        self.assertGreater(y, btn.y() + btn.height())

    def test_tour_positions_panel_away_from_open_dropdown(self) -> None:
        from PyQt6.QtWidgets import QMenu, QPushButton, QWidget

        from ui.components.onboarding_tour import OnboardingStep, OnboardingTour

        host = QWidget()
        host.resize(900, 700)
        host.show()

        btn = QPushButton("Filter", host)
        btn.move(200, 24)
        btn.resize(120, 32)
        btn.show()
        menu = QMenu(btn)
        menu.addAction("Option A")
        menu.addAction("Option B")
        btn.setMenu(menu)
        btn.showMenu()

        tour = OnboardingTour(
            host,
            [
                OnboardingStep(
                    step_id="dropdown",
                    title="Filter",
                    body="Choose a filter option from this menu.",
                    target_getter=lambda _h: btn,
                )
            ],
        )
        tour.start()
        self._app.processEvents()

        panel_top = tour._panel.y()
        panel_bottom = panel_top + tour._panel.height()
        menu_top = host.mapFromGlobal(menu.mapToGlobal(menu.rect().topLeft())).y()
        menu_bottom = menu_top + menu.height()
        overlaps_menu = panel_bottom > menu_top and panel_top < menu_bottom
        self.assertFalse(overlaps_menu)
        tour.finish()

    def test_next_closes_dropdown_and_advances(self) -> None:
        from PyQt6.QtWidgets import QMenu, QPushButton, QWidget

        from ui.components.onboarding_tour import OnboardingStep, OnboardingTour

        host = QWidget()
        host.resize(900, 700)
        host.show()

        btn = QPushButton("Filter", host)
        btn.move(200, 24)
        btn.resize(120, 32)
        btn.show()
        menu = QMenu(btn)
        menu.addAction("Option A")
        menu.addAction("Option B")
        btn.setMenu(menu)
        btn.showMenu()

        tour = OnboardingTour(
            host,
            [
                OnboardingStep(
                    step_id="dropdown",
                    title="Filter",
                    body="Choose a filter option from this menu.",
                    target_getter=lambda _h: btn,
                ),
                OnboardingStep(
                    step_id="done",
                    title="Done",
                    body="Moved on.",
                ),
            ],
        )
        tour.start()
        self._app.processEvents()
        self.assertEqual(tour._index, 0)
        self.assertIsNotNone(self._app.activePopupWidget())

        tour.next()
        self._app.processEvents()

        self.assertIsNone(self._app.activePopupWidget())
        self.assertEqual(tour._index, 1)
        tour.finish()

    def test_scroll_target_into_view_reveals_offscreen_widget(self) -> None:
        from PyQt6.QtCore import QPoint, QRect
        from PyQt6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

        host = QWidget()
        host.resize(420, 260)
        host.show()

        scroll = QScrollArea(host)
        scroll.setWidgetResizable(True)
        scroll.resize(420, 260)
        scroll.move(0, 0)
        scroll.show()

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.addStretch(1)
        target = QFrame()
        target.setFixedHeight(80)
        layout.addWidget(target)
        layout.addStretch(1)
        content.setMinimumHeight(900)

        self._app.processEvents()
        bar = scroll.verticalScrollBar()
        bar.setValue(0)
        self._app.processEvents()

        vp = scroll.viewport()
        target_rect = QRect(target.mapTo(vp, QPoint(0, 0)), target.size())
        self.assertFalse(vp.rect().contains(target_rect))

        _scroll_target_into_view(target)
        self._app.processEvents()

        target_rect = QRect(target.mapTo(vp, QPoint(0, 0)), target.size())
        self.assertTrue(vp.rect().contains(target_rect))

    def test_tour_scrolls_offscreen_step_target_into_view(self) -> None:
        from PyQt6.QtCore import QPoint, QRect
        from PyQt6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

        from ui.components.onboarding_tour import OnboardingStep, OnboardingTour

        host = QWidget()
        host.resize(420, 260)
        host.show()

        scroll = QScrollArea(host)
        scroll.setWidgetResizable(True)
        scroll.resize(420, 260)
        scroll.move(0, 0)
        scroll.show()

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.addStretch(1)
        target = QFrame()
        target.setFixedHeight(80)
        layout.addWidget(target)
        layout.addStretch(1)
        content.setMinimumHeight(900)

        self._app.processEvents()
        scroll.verticalScrollBar().setValue(0)
        self._app.processEvents()

        tour = OnboardingTour(
            host,
            [
                OnboardingStep(
                    step_id="offscreen",
                    title="Below the fold",
                    body="This card should scroll into view automatically.",
                    target_getter=lambda _h: target,
                )
            ],
        )
        tour.start()
        self._app.processEvents()

        vp = scroll.viewport()
        target_rect = QRect(target.mapTo(vp, QPoint(0, 0)), target.size())
        self.assertTrue(vp.rect().contains(target_rect))
        tour.finish()


if __name__ == "__main__":
    unittest.main()
