"""Unified per-turn execution context."""
from __future__ import annotations

import unittest

from core.discourse_intent import FollowUpClassification, FollowUpKind
from core.turn_context import resolve_history_strategy, resolve_turn_context


class TestTurnContext(unittest.TestCase):
    def test_native_roles_for_non_harmony(self) -> None:
        self.assertEqual(resolve_history_strategy(use_harmony_protocol=False), "native_roles")

    def test_harmony_compact_for_harmony(self) -> None:
        self.assertEqual(resolve_history_strategy(use_harmony_protocol=True), "harmony_compact")

    def test_resolve_turn_context_trace_fields(self) -> None:
        fu = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.7, ("anaphoric",))
        ctx = resolve_turn_context(
            execution_route="NONE",
            user_query="And its minorities?",
            follow_up=fu,
            prior_turn_unreliable=True,
            history_turn_count=7,
            use_harmony_protocol=True,
        )
        fields = ctx.trace_fields()
        self.assertEqual(fields["history_strategy"], "harmony_compact")
        self.assertIn("chat_format_mode", fields)
        self.assertIn("generation_risk_tier", fields)


if __name__ == "__main__":
    unittest.main()
