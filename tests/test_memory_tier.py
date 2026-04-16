"""Unit tests for ``core.memory_filters.derive_memory_tier`` (T3.4).

Exhaustive table test of the classification rules in §3.2 of
``docs/memory_enrichment_T3_2_T3_4_plan.md``. The mapping is a pure
function from fact payload -> structural tier, so we can cover every
row of the table without touching LanceDB, PyQt, or the LLM.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.memory_filters import MEMORY_TIERS, derive_memory_tier


class DeriveMemoryTierTests(unittest.TestCase):
    # -------------------- episode --------------------

    def test_episode_category_always_episode(self):
        """Rule 1: category=="episode" beats every other signal."""
        fact = {
            "category": "episode",
            "subject": "user",
            "origin": "user_stated",
        }
        self.assertEqual(derive_memory_tier(fact), "episode")

    def test_episode_category_beats_third_party_subject(self):
        fact = {
            "category": "episode",
            "subject": "third_party",
            "origin": "document_derived",
        }
        self.assertEqual(derive_memory_tier(fact), "episode")

    # -------------------- preference --------------------

    def test_user_stated_preference(self):
        """Rule 2: subject=user AND origin=user_stated -> preference."""
        fact = {
            "category": "preference",
            "subject": "user",
            "origin": "user_stated",
        }
        self.assertEqual(derive_memory_tier(fact), "preference")

    def test_user_confirmed_preference(self):
        """Rule 2: user_confirmed also lands in preference."""
        fact = {
            "category": "identity",
            "subject": "user",
            "origin": "user_confirmed",
        }
        self.assertEqual(derive_memory_tier(fact), "preference")

    def test_user_subject_but_document_origin_is_context(self):
        """User subject without user_stated/user_confirmed falls through."""
        fact = {
            "category": "context",
            "subject": "user",
            "origin": "document_derived",
        }
        # subject=user but origin is document_derived -> falls through
        # to the "document_derived -> knowledge" rule.
        self.assertEqual(derive_memory_tier(fact), "knowledge")

    # -------------------- knowledge --------------------

    def test_explicit_remember_flag_forces_knowledge(self):
        """Rule 3: _explicit_remember trumps subject/origin heuristics."""
        fact = {
            "category": "context",
            "subject": "third_party",
            "origin": "user_stated",
            "_explicit_remember": True,
        }
        self.assertEqual(derive_memory_tier(fact), "knowledge")

    def test_third_party_subject_knowledge(self):
        """Rule 4: subject=third_party -> knowledge."""
        fact = {
            "category": "knowledge",
            "subject": "third_party",
            "origin": "user_stated",
        }
        self.assertEqual(derive_memory_tier(fact), "knowledge")

    def test_document_derived_origin_knowledge(self):
        """Rule 5: origin=document_derived -> knowledge."""
        fact = {
            "category": "context",
            "subject": "unknown",
            "origin": "document_derived",
        }
        self.assertEqual(derive_memory_tier(fact), "knowledge")

    # -------------------- context --------------------

    def test_unclassified_falls_through_to_context(self):
        """Rule 6: nothing matched -> context."""
        fact = {
            "category": "context",
            "subject": "unknown",
            "origin": "inferred",
        }
        self.assertEqual(derive_memory_tier(fact), "context")

    def test_missing_keys_returns_context(self):
        """Defensive: empty dict is safe."""
        self.assertEqual(derive_memory_tier({}), "context")

    def test_none_input_returns_context(self):
        """Defensive: non-dict inputs are safe."""
        self.assertEqual(derive_memory_tier(None), "context")       # type: ignore[arg-type]
        self.assertEqual(derive_memory_tier("fact"), "context")    # type: ignore[arg-type]
        self.assertEqual(derive_memory_tier(42), "context")         # type: ignore[arg-type]

    # -------------------- export / constants --------------------

    def test_memory_tiers_constant_has_four_tiers(self):
        self.assertEqual(
            tuple(sorted(MEMORY_TIERS)),
            ("context", "episode", "knowledge", "preference"),
        )


if __name__ == "__main__":
    unittest.main()
