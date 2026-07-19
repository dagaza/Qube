"""Smoke tests for the Advanced Telemetry page guided tour."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import build_tour


class TestTelemetryTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self):
        from PyQt6.QtWidgets import QFrame, QPushButton, QWidget

        host = QWidget()
        tv = QWidget(host)
        host.nav_telemetry = QPushButton(host)
        host._telemetry_view = tv
        host.ensure_telemetry_view = MagicMock(return_value=tv)
        host._route_view = MagicMock()

        tv.hardware_card = QFrame(tv)
        tv.latency_card = QFrame(tv)
        tv.model_capability_card = QFrame(tv)
        tv.router_card = QFrame(tv)
        tv.sidecar_card = QFrame(tv)
        tv.inference_transparency_card = QFrame(tv)

        return host

    def test_step_count_matches_plan_blocks(self) -> None:
        tour = build_tour("telemetry", self._make_host())
        self.assertIsNotNone(tour)
        assert tour is not None
        # welcome + 6 cards + 1 finish = 8
        self.assertEqual(len(tour._steps), 8)

    def test_step_order_section_anchors(self) -> None:
        tour = build_tour("telemetry", self._make_host())
        assert tour is not None
        ids = [step.step_id for step in tour._steps]
        self.assertEqual(ids[0], "welcome")
        self.assertEqual(
            ids[1:7],
            [
                "hardware",
                "latency",
                "capability",
                "router",
                "sidecar",
                "inference_stack",
            ],
        )
        self.assertEqual(ids[-1], "tour_complete")

    def test_all_target_getters_resolve(self) -> None:
        host = self._make_host()
        tour = build_tour("telemetry", host)
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

    def test_on_enter_routes_to_telemetry(self) -> None:
        host = self._make_host()
        tour = build_tour("telemetry", host)
        assert tour is not None
        tour._steps[0].on_enter(host)
        host._route_view.assert_called_with(3, host.nav_telemetry)


if __name__ == "__main__":
    unittest.main()
