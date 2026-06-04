"""Tests for companion motif rotation."""

from __future__ import annotations

import unittest

from core.companion_cognition.motifs import (
    MOTIF_CATALOG,
    MotifState,
    motif_recent_penalty,
    motif_selection_boost,
    resolve_active_motif,
)


class TestCompanionMotifs(unittest.TestCase):
    def test_weekly_rotation_stable(self) -> None:
        ts = 1_735_000_000.0
        state = MotifState(active_motif="pixels", motif_since_ts=1.0)
        a = resolve_active_motif(state, ts)
        b = resolve_active_motif(MotifState(active_motif="pixels", motif_since_ts=1.0), ts)
        self.assertEqual(a, b)
        self.assertIn(a, MOTIF_CATALOG)

    def test_motif_boost_only_when_match(self) -> None:
        self.assertGreater(motif_selection_boost("observing", ("observing",)), 1.0)
        self.assertEqual(motif_selection_boost("observing", ("pixels",)), 1.0)

    def test_recent_motif_penalty(self) -> None:
        recent = ("quiet", "quiet", "quiet")
        self.assertLess(motif_recent_penalty(recent, ("quiet",)), 1.0)


if __name__ == "__main__":
    unittest.main()
