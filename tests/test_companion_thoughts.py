"""Tests for companion thought engine."""

from __future__ import annotations

import time
import unittest

from core.companion_cognition.personality import vector_from_trait_preset
from core.companion_cognition.thoughts import derive_voice, think
from core.companion_cognition.types import CompanionObservation
from core.companion_cognition.variety import VarietySnapshot
from core.companion_verbal_traits import CompanionVerbalTraitPreset


class TestCompanionThoughts(unittest.TestCase):
    def test_library_update_prefers_acknowledge_or_celebration(self) -> None:
        obs = CompanionObservation(
            type="library_update_completed",
            facts={"file_count": 2},
            trigger_source="ingest_complete",
        )
        personality = vector_from_trait_preset(CompanionVerbalTraitPreset.WARM)
        variety = VarietySnapshot(now=time.time())
        thought = think(obs, personality, variety)
        self.assertIsNotNone(thought)
        assert thought is not None
        self.assertIn(thought.intent, ("acknowledge_effort",))
        self.assertIn("file_count_word", thought.slots)

    def test_quiet_period_returns_wellbeing_family(self) -> None:
        obs = CompanionObservation(type="quiet_period", facts={"idle_sec": 90.0, "main_hidden": True})
        personality = vector_from_trait_preset(CompanionVerbalTraitPreset.NEUTRAL)
        variety = VarietySnapshot(now=time.time())
        thought = think(obs, personality, variety)
        self.assertIsNotNone(thought)
        assert thought is not None
        self.assertIn(
            thought.intent,
            (
                "wellbeing",
                "atmosphere",
                "self_expression",
                "curiosity",
                "reflection",
                "humor",
                "fact",
            ),
        )
        self.assertTrue(thought.voice)

    def test_derive_voice_for_humor(self) -> None:
        personality = vector_from_trait_preset(CompanionVerbalTraitPreset.WITTY)
        self.assertEqual(derive_voice("humor", personality), "dry")
        neutral = vector_from_trait_preset(CompanionVerbalTraitPreset.NEUTRAL)
        self.assertEqual(derive_voice("humor", neutral), "playful")


if __name__ == "__main__":
    unittest.main()
