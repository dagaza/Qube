"""Generation risk profile for collapse-prone turns."""
from __future__ import annotations

import unittest

from core.generation_risk_profile import resolve_generation_risk_profile


class TestGenerationRiskProfile(unittest.TestCase):
    def test_low_risk_simple_lookup(self) -> None:
        profile = resolve_generation_risk_profile(
            user_query="What is the capital of France?",
            chat_format_mode="brief",
        )
        self.assertEqual(profile.risk_tier, "low")
        self.assertEqual(profile.temperature_multiplier, 1.0)

    def test_high_risk_after_unreliable_prior_and_list(self) -> None:
        profile = resolve_generation_risk_profile(
            user_query="List the major ethnic groups",
            chat_format_mode="structured",
            prior_turn_unreliable=True,
            history_turn_count=8,
            require_list_format=True,
            follow_up_active=True,
        )
        self.assertEqual(profile.risk_tier, "high")
        self.assertLess(profile.temperature_multiplier, 1.0)
        self.assertGreater(profile.repeat_penalty_adjust, 0.0)
        self.assertTrue(profile.enable_list_loop_guard)
        self.assertEqual(profile.stream_guard_min_repeats, 6)

    def test_effective_temperature_respects_bounds(self) -> None:
        profile = resolve_generation_risk_profile(
            user_query="List items",
            chat_format_mode="structured",
            prior_turn_unreliable=True,
            require_list_format=True,
            history_turn_count=10,
            follow_up_active=True,
        )
        self.assertGreaterEqual(profile.effective_temperature(0.7), 0.05)
        self.assertLessEqual(profile.effective_temperature(0.7), 2.0)


if __name__ == "__main__":
    unittest.main()
