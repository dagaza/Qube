"""Onboarding coach panel layout — wrapped body text must not clip."""

from __future__ import annotations

import unittest

from core.catalog_hardware_recommendation import build_tour_model_download_body
from core.hardware_capability_profile import HardwareCapabilityProfile, HardwareTier
from tests.test_catalog_hardware_recommendation import _entry
from ui.components.onboarding_tour import OnboardingCoachPanel


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


if __name__ == "__main__":
    unittest.main()
