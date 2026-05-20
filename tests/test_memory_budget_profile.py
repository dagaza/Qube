"""Tests for soft memory budget experience labels."""

from __future__ import annotations

import unittest
from unittest import mock

from core.memory_budget_profile import (
    MemoryBudgetKind,
    MemoryBudgetProfile,
    experience_for_download,
)


class TestMemoryBudgetProfile(unittest.TestCase):
    def test_unknown_budget_no_fit_wording(self) -> None:
        exp = experience_for_download(
            5_000_000_000,
            MemoryBudgetProfile(kind=MemoryBudgetKind.UNKNOWN, budget_bytes=0),
        )
        self.assertNotIn("Does not fit", exp.short_label)
        self.assertNotIn("Does not fit", exp.detail)
        self.assertIn("File", exp.short_label)

    def test_dedicated_under_budget_best(self) -> None:
        exp = experience_for_download(
            4_000_000_000,
            MemoryBudgetProfile(kind=MemoryBudgetKind.DEDICATED_VRAM, budget_bytes=8_000_000_000),
        )
        self.assertIn("Best responsiveness", exp.short_label)
        self.assertEqual(exp.style, "best")

    def test_dedicated_over_budget_soft_wording(self) -> None:
        exp = experience_for_download(
            10_000_000_000,
            MemoryBudgetProfile(kind=MemoryBudgetKind.DEDICATED_VRAM, budget_bytes=4_000_000_000),
        )
        self.assertNotIn("Does not fit", exp.short_label)
        self.assertIn("shared", exp.detail.lower())

    def test_unified_profile(self) -> None:
        exp = experience_for_download(
            3_000_000_000,
            MemoryBudgetProfile(kind=MemoryBudgetKind.UNIFIED, budget_bytes=16_000_000_000),
        )
        self.assertIn("Unified memory", exp.short_label)

    @mock.patch("core.memory_budget_profile._is_apple_unified", return_value=True)
    @mock.patch("core.memory_budget_profile.detect_gpu_vram_bytes", return_value=16_000_000_000)
    def test_detect_unified_on_apple(self, _v: mock.Mock, _a: mock.Mock) -> None:
        from core.memory_budget_profile import detect_memory_budget_profile

        p = detect_memory_budget_profile()
        self.assertEqual(p.kind, MemoryBudgetKind.UNIFIED)


if __name__ == "__main__":
    unittest.main()
