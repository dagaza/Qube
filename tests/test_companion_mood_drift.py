"""Tests for ambient mood drift."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.companion_cognition.mood_drift import (
    AMBIENT_MOOD_STATES,
    AmbientMoodState,
    ambient_mood_intent_bias,
    load_mood_state,
    tick_mood_drift,
)
from core.companion_cognition.personality import CompanionPersonalityVector
from core.companion_verbal_traits import CompanionVerbalTraitPreset
from core.companion_cognition.personality import vector_from_trait_preset


class TestCompanionMoodDrift(unittest.TestCase):
    def test_tick_major_drift_is_deterministic(self) -> None:
        personality = vector_from_trait_preset(CompanionVerbalTraitPreset.NEUTRAL)
        start = AmbientMoodState(state="observant", strength=0.5, last_drift_ts=1_000_000.0)
        now = start.last_drift_ts + 13 * 3600
        a = tick_mood_drift(start, personality, now_ts=now, on_session_start=True)
        b = tick_mood_drift(start, personality, now_ts=now, on_session_start=True)
        self.assertEqual(a.state, b.state)
        self.assertAlmostEqual(a.strength, b.strength)
        self.assertIn(a.state, AMBIENT_MOOD_STATES)

    def test_strength_clamped(self) -> None:
        s = AmbientMoodState(state="cozy", strength=2.0).clamped()
        self.assertLessEqual(s.strength, 0.85)
        self.assertGreaterEqual(s.strength, 0.35)

    def test_ambient_intent_bias_never_zero(self) -> None:
        for state in AMBIENT_MOOD_STATES:
            self.assertGreaterEqual(ambient_mood_intent_bias(state, "wellbeing"), 1.0)

    def test_load_default_when_missing(self) -> None:
        with patch("core.companion_cognition.mood_drift._MOOD_PATH", Path(tempfile.gettempdir()) / "missing_mood.json"):
            s = load_mood_state()
            self.assertIn(s.state, AMBIENT_MOOD_STATES)


if __name__ == "__main__":
    unittest.main()
