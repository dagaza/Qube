"""Companion caption line quality gates."""
from __future__ import annotations

import unittest

from core.companion_line_quality import is_acceptable_companion_line


class TestCompanionLineQuality(unittest.TestCase):
    def test_accepts_user_facing_line(self) -> None:
        self.assertTrue(is_acceptable_companion_line("Still here if you need me."))

    def test_rejects_placeholder_about_companion(self) -> None:
        self.assertFalse(is_acceptable_companion_line("Maybe something about the companion"))

    def test_rejects_meta_tutorial(self) -> None:
        self.assertFalse(
            is_acceptable_companion_line(
                "Welcome to the Qube desktop companion, customize your settings"
            )
        )


if __name__ == "__main__":
    unittest.main()
