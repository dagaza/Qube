"""Inference-time discourse query rewriting."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import classify_follow_up  # noqa: E402
from core.discourse_query_rewrite import (  # noqa: E402
    REWRITE_CONFIDENCE_MIN,
    resolve_ambiguous_user_query,
)
from core.discourse_state import (  # noqa: E402
    DiscourseState,
    promote_referent_after_assistant,
    update_discourse_state,
)


class TestDiscourseQueryRewrite(unittest.TestCase):
    def _kathmandu_state(self) -> DiscourseState:
        prior = update_discourse_state(
            [{"role": "user", "content": "What is the capital of Nepal?"}],
            None,
            "What is the capital of Nepal?",
        )
        return promote_referent_after_assistant(
            user_prompt="What is the capital of Nepal?",
            assistant_text="Kathmandu is the capital of Nepal.",
            prior=prior,
        )

    def test_its_population_rewrites_to_kathmandu(self) -> None:
        state = self._kathmandu_state()
        prompt = "And what is the size of its population?"
        history = [
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
            {"role": "user", "content": prompt},
        ]
        follow_up = classify_follow_up(prompt, history, state)
        resolved = resolve_ambiguous_user_query(prompt, state, follow_up)
        self.assertTrue(resolved.succeeded)
        self.assertGreaterEqual(resolved.confidence, REWRITE_CONFIDENCE_MIN)
        self.assertIn("Kathmandu", resolved.resolved)

    def test_possessive_area_rewrite(self) -> None:
        state = self._kathmandu_state()
        prompt = "what is its area"
        follow_up = classify_follow_up(prompt, [], state)
        resolved = resolve_ambiguous_user_query(prompt, state, follow_up)
        self.assertIn("Kathmandu", resolved.resolved)

    def test_surface_area_follow_up_rewrites_to_entity(self) -> None:
        state = promote_referent_after_assistant(
            user_prompt="What is the capital of Romania?",
            assistant_text="Bucharest is the capital of Romania.",
            prior=update_discourse_state(
                [{"role": "user", "content": "What is the capital of Romania?"}],
                None,
                "What is the capital of Romania?",
            ),
        )
        prompt = "Great, and how about its surface area?"
        follow_up = classify_follow_up(prompt, [], state)
        resolved = resolve_ambiguous_user_query(prompt, state, follow_up)
        self.assertEqual(resolved.resolved, "What is the surface area of Bucharest?")

    def test_he_born_rewrite_with_person_referent(self) -> None:
        state = DiscourseState(
            active_referent="Steve Jobs",
            referent_type="person",
            confidence=0.85,
            referent_confidence=0.85,
        )
        prompt = "when was he born"
        follow_up = classify_follow_up(prompt, [{"role": "user", "content": "Who founded Apple?"}], state)
        resolved = resolve_ambiguous_user_query(prompt, state, follow_up)
        self.assertIn("Steve Jobs", resolved.resolved)

    def test_no_rewrite_without_referent(self) -> None:
        follow_up = classify_follow_up("what is its population", [], None)
        resolved = resolve_ambiguous_user_query("what is its population", None, follow_up)
        self.assertFalse(resolved.succeeded)
        self.assertEqual(resolved.resolved, resolved.original)

    def test_no_rewrite_when_entity_already_named(self) -> None:
        state = DiscourseState(
            active_referent="Kathmandu",
            referent_type="city",
            confidence=0.9,
        )
        prompt = "What is the population of Kathmandu?"
        follow_up = classify_follow_up(prompt, [], state)
        resolved = resolve_ambiguous_user_query(prompt, state, follow_up)
        self.assertFalse(resolved.succeeded)


if __name__ == "__main__":
    unittest.main()
