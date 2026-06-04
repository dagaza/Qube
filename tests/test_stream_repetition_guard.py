from __future__ import annotations

import unittest

from core.stream_repetition_guard import StreamRepetitionGuard


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


if __name__ == "__main__":
    unittest.main()
