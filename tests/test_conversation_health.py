"""Conversation health escalation across turns."""
from __future__ import annotations

import unittest

from core.conversation_health import (
    TurnAnomalyOutcome,
    initial_conversation_health,
    merge_generation_risk_with_health,
    resolve_conversation_health_policy,
    update_conversation_health,
)
from core.generation_risk_profile import resolve_generation_risk_profile


class TestConversationHealth(unittest.TestCase):
    def test_initial_health_is_normal(self) -> None:
        state = initial_conversation_health()
        self.assertEqual(state.health_score, 1.0)
        self.assertEqual(state.mode, "normal")

    def test_high_anomaly_drops_health_by_035(self) -> None:
        before = initial_conversation_health()
        after = update_conversation_health(
            before,
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        self.assertAlmostEqual(after.health_score, 0.65)
        self.assertEqual(after.mode, "warning")
        self.assertEqual(after.consecutive_anomalies, 1)

    def test_consecutive_anomaly_escalates_further(self) -> None:
        turn7 = update_conversation_health(
            initial_conversation_health(),
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        turn8 = update_conversation_health(
            turn7,
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        self.assertLess(turn8.health_score, turn7.health_score)
        self.assertEqual(turn8.mode, "recovery")

    def test_clean_turn_recovers_slightly(self) -> None:
        degraded = update_conversation_health(
            initial_conversation_health(),
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        recovered = update_conversation_health(
            degraded,
            outcome=TurnAnomalyOutcome(degeneration_risk="LOW"),
        )
        self.assertGreater(recovered.health_score, degraded.health_score)
        self.assertEqual(recovered.consecutive_anomalies, 0)

    def test_warning_policy_disables_discourse_rewrite(self) -> None:
        state = update_conversation_health(
            initial_conversation_health(),
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        policy = resolve_conversation_health_policy(state)
        self.assertEqual(policy.mode, "warning")
        self.assertFalse(policy.allow_discourse_rewrite)
        self.assertFalse(policy.allow_query_rewrite)
        self.assertLess(policy.temperature_multiplier, 1.0)

    def test_recovery_policy_is_strictest(self) -> None:
        state = update_conversation_health(
            initial_conversation_health(),
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        state = update_conversation_health(
            state,
            outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
        )
        policy = resolve_conversation_health_policy(state)
        self.assertEqual(policy.mode, "recovery")
        self.assertEqual(policy.max_tokens_cap, 1024)
        self.assertEqual(policy.stream_guard_min_repeats, 4)

    def test_merge_generation_risk_with_health(self) -> None:
        base = resolve_generation_risk_profile(
            user_query="hello",
            chat_format_mode="brief",
        )
        policy = resolve_conversation_health_policy(
            update_conversation_health(
                initial_conversation_health(),
                outcome=TurnAnomalyOutcome(degeneration_risk="HIGH"),
            )
        )
        merged = merge_generation_risk_with_health(base, policy)
        self.assertLess(merged.temperature_multiplier, base.temperature_multiplier)
        self.assertIn("conversation_health_warning", merged.signals)


if __name__ == "__main__":
    unittest.main()
