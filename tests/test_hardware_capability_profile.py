"""Tests for hardware capability profile detection."""

from __future__ import annotations

import unittest

from core.hardware_capability_profile import (
    HardwareCapabilityProfile,
    HardwareTier,
    classify_hardware_tier,
    detect_hardware_capability_profile,
)


class TestHardwareCapabilityProfile(unittest.TestCase):
    def test_classify_compact_vram(self) -> None:
        self.assertEqual(
            classify_hardware_tier(vram_gb=4.0, ram_gb=16.0),
            HardwareTier.COMPACT,
        )

    def test_classify_standard_vram(self) -> None:
        self.assertEqual(
            classify_hardware_tier(vram_gb=8.0, ram_gb=16.0),
            HardwareTier.STANDARD,
        )

    def test_classify_cpu_only_uses_ram(self) -> None:
        self.assertEqual(
            classify_hardware_tier(vram_gb=0.0, ram_gb=64.0),
            HardwareTier.PERFORMANCE,
        )

    def test_inference_budget_prefers_vram(self) -> None:
        profile = HardwareCapabilityProfile(
            total_ram_gb=32.0,
            total_vram_gb=8.0,
            cpu_cores=8,
            gpu_name="Test GPU",
            gpu_backend="nvidia",
            tier=HardwareTier.STANDARD,
        )
        self.assertAlmostEqual(profile.inference_budget_gb, 6.8)

    def test_detect_returns_profile(self) -> None:
        profile = detect_hardware_capability_profile()
        self.assertGreaterEqual(profile.total_ram_gb, 0.0)
        self.assertGreaterEqual(profile.cpu_cores, 1)


if __name__ == "__main__":
    unittest.main()
