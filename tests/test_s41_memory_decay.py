"""Plan S41 (V67, D45): knowledge memories do not decay.

Recall ranks memories by score times a temporal-decay multiplier. With a
30-day half-life on the knowledge tier (facts about other people, explicitly
remembered facts, document-derived facts), any recent note outranked an older
fact ("What's my cat called?": the cat memory was 160 days old). Knowledge is
now evergreen, like preferences; context notes and episodes still decay. The
first two tests fail on the old half-life.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.memory_filters import derive_memory_tier  # noqa: E402
from core.memory_retrieval_policy import TEMPORAL_HALF_LIFE_DAYS, temporal_decay_multiplier  # noqa: E402

NOW = 1_800_000_000.0
DAY = 86400.0


def _payload(age_days: float) -> dict:
    return {"last_used_at": NOW - age_days * DAY}


class TestTemporalDecayTiers(unittest.TestCase):
    def test_knowledge_tier_is_evergreen(self):
        self.assertIsNone(TEMPORAL_HALF_LIFE_DAYS["knowledge"])
        for age in (0, 30, 160, 400):
            self.assertEqual(temporal_decay_multiplier("knowledge", _payload(age), now=NOW), 1.0)

    def test_facts_that_no_longer_decay(self):
        # Which memories D45 affects: the knowledge tier, by subject and origin.
        old_fact = _payload(160)
        for fact in (
            {"category": "knowledge", "subject": "third_party", "origin": "user_stated"},
            {"category": "identity", "subject": "third_party", "origin": "user_stated"},
            {"category": "knowledge", "subject": "third_party", "origin": "document_derived"},
            {"category": "context", "subject": "user", "origin": "system_derived", "_explicit_remember": True},
        ):
            with self.subTest(fact=fact):
                tier = derive_memory_tier(fact)
                self.assertEqual(tier, "knowledge")
                self.assertEqual(temporal_decay_multiplier(tier, old_fact, now=NOW), 1.0)

    def test_tiers_that_still_decay_or_never_did(self):
        self.assertEqual(derive_memory_tier({"category": "identity", "subject": "user", "origin": "user_stated"}), "preference")
        self.assertEqual(derive_memory_tier({"category": "project", "subject": "user", "origin": "system_derived"}), "context")
        self.assertEqual(derive_memory_tier({"category": "episode"}), "episode")

    def test_preference_still_evergreen(self):
        self.assertEqual(temporal_decay_multiplier("preference", _payload(200), now=NOW), 1.0)

    def test_context_and_episode_still_decay(self):
        self.assertAlmostEqual(temporal_decay_multiplier("context", _payload(30), now=NOW), 0.5, places=6)
        self.assertAlmostEqual(temporal_decay_multiplier("episode", _payload(14), now=NOW), 0.5, places=6)
        self.assertLess(temporal_decay_multiplier("context", _payload(160), now=NOW), 0.03)


if __name__ == "__main__":
    unittest.main()
