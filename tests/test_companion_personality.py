"""Tests for companion personality vector."""

from __future__ import annotations

import unittest

from core.companion_cognition.personality import (
    CompanionPersonalityVector,
    vector_from_trait_preset,
)
from core.companion_verbal_traits import CompanionVerbalTraitPreset


class TestCompanionPersonality(unittest.TestCase):
    def test_trait_preset_migration(self) -> None:
        warm = vector_from_trait_preset(CompanionVerbalTraitPreset.WARM)
        self.assertGreaterEqual(warm.warmth, 0.8)

    def test_vector_clamps(self) -> None:
        v = CompanionPersonalityVector(warmth=1.5, humor=-0.2).clamped()
        self.assertEqual(v.warmth, 1.0)
        self.assertEqual(v.humor, 0.0)


if __name__ == "__main__":
    unittest.main()
