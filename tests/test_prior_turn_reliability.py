"""Prior turn reliability hints after history suppression."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.history_degeneration import HISTORY_SUPPRESSION_PLACEHOLDER  # noqa: E402
from core.prior_turn_reliability import (  # noqa: E402
    build_prior_turn_unreliable_suffix,
    history_contains_suppressed_assistant,
)


class TestPriorTurnReliability(unittest.TestCase):
    def test_detects_suppressed_assistant(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": HISTORY_SUPPRESSION_PLACEHOLDER},
            {"role": "user", "content": "follow up"},
        ]
        self.assertTrue(history_contains_suppressed_assistant(history))

    def test_ignores_normal_assistant(self) -> None:
        history = [
            {"role": "assistant", "content": "Kathmandu is the capital."},
            {"role": "user", "content": "follow up"},
        ]
        self.assertFalse(history_contains_suppressed_assistant(history))

    def test_suffix_nonempty(self) -> None:
        self.assertIn("unreliable", build_prior_turn_unreliable_suffix().lower())


if __name__ == "__main__":
    unittest.main()
