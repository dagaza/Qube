from __future__ import annotations

import unittest

from core.stream_repetition_guard import StreamRepetitionGuard, create_stream_repetition_guard
from core.generation_risk_profile import resolve_generation_risk_profile


class StreamRepetitionGuardTests(unittest.TestCase):
    def test_trips_on_spaced_unicode_punctuation_tail(self) -> None:
        guard = StreamRepetitionGuard()
        self.assertFalse(
            guard.observe(
                "Got it—he wasn’t talking about Microsoft Cloud Platform. "
            )
        )
        self.assertTrue(
            guard.observe(
                "In that context MCP is just another way of saying the … … "
                "……‑…‑…‑…‑……… … … ‑ …‑… … …………"
            )
        )
        self.assertEqual(
            guard.trip_reason,
            "spaced punctuation degeneration in stream tail",
        )

    def test_does_not_trip_on_normal_short_punctuation(self) -> None:
        guard = StreamRepetitionGuard()
        self.assertFalse(guard.observe("Wait... are you sure? Yes—probably."))

    def test_list_loop_guard_trips_on_empty_numbered_lines(self) -> None:
        guard = StreamRepetitionGuard(enable_list_loop_guard=True, min_repeats=20)
        payload = "Groups include:\n1.\n2.\n3.\n4.\n5."
        self.assertTrue(guard.observe(payload))
        self.assertEqual(
            guard.trip_reason,
            "numbered list loop degeneration in stream tail",
        )

    def test_create_from_risk_profile_enables_list_guard(self) -> None:
        profile = resolve_generation_risk_profile(
            user_query="List major sites",
            chat_format_mode="structured",
            require_list_format=True,
            prior_turn_unreliable=True,
            history_turn_count=8,
        )
        guard = create_stream_repetition_guard(profile)
        self.assertTrue(guard.observe("1.\n2.\n3.\n4.\n"))


if __name__ == "__main__":
    unittest.main()
