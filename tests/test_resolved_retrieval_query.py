"""Canonical ResolvedRetrievalQuery builder and web inference alignment."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import classify_follow_up  # noqa: E402
from core.discourse_query import (  # noqa: E402
    build_resolved_retrieval_query,
    resolve_search_target,
)
from core.discourse_query_rewrite import (  # noqa: E402
    REWRITE_CONFIDENCE_MIN,
    resolve_ambiguous_user_query,
)
from core.discourse_state import (  # noqa: E402
    promote_referent_after_assistant,
    update_discourse_state,
)


class TestResolvedRetrievalQuery(unittest.TestCase):
    def _kathmandu_state(self):
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

    def test_web_uses_inference_text_not_raw(self) -> None:
        state = self._kathmandu_state()
        raw = "And what is the size of its population?"
        history = [
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
            {"role": "user", "content": raw},
        ]
        follow_up = classify_follow_up(raw, history, state)
        resolved_query = resolve_ambiguous_user_query(raw, state, follow_up)
        self.assertTrue(resolved_query.succeeded)
        self.assertGreaterEqual(resolved_query.confidence, REWRITE_CONFIDENCE_MIN)

        canonical = build_resolved_retrieval_query(
            raw_text=raw,
            inference_text=resolved_query.resolved,
            follow_up=follow_up,
            discourse=state,
            history=history,
            resolved_query=resolved_query,
        )

        legacy_raw_web = resolve_search_target(
            raw, follow_up, state, history
        ).query
        self.assertIn("its", legacy_raw_web.lower())
        self.assertIn("Kathmandu", canonical.web_text)
        self.assertNotIn(" its ", f" {canonical.web_text.lower()} ")

    def test_routing_and_retrieval_share_inference_base(self) -> None:
        state = self._kathmandu_state()
        raw = "What about its music?"
        follow_up = classify_follow_up(raw, [], state)
        inference = "What about Kathmandu's music?"
        canonical = build_resolved_retrieval_query(
            raw_text=raw,
            inference_text=inference,
            follow_up=follow_up,
            discourse=state,
            history=[],
        )
        self.assertEqual(canonical.routing_text, canonical.retrieval_text)
        self.assertIn("Kathmandu", canonical.routing_text)

    def test_meta_web_prior_turn_uses_history(self) -> None:
        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {"role": "assistant", "content": "Birds dust-bathe to stay clean."},
            {
                "role": "user",
                "content": (
                    "Yes that would be nice. Can you also do an online search "
                    "for the answer?"
                ),
            },
        ]
        raw = history[-1]["content"]
        follow_up = classify_follow_up(raw, history, None)
        state = update_discourse_state(history, None, raw)
        canonical = build_resolved_retrieval_query(
            raw_text=raw,
            inference_text=raw,
            follow_up=follow_up,
            discourse=state,
            history=history,
        )
        self.assertEqual(canonical.web_text, "Why do birds take dust baths?")
        self.assertEqual(canonical.web_rewrite_reason, "meta_prior_turn")

    def test_telemetry_dict_includes_web_and_inference(self) -> None:
        state = self._kathmandu_state()
        raw = "And what is the size of its population?"
        follow_up = classify_follow_up(raw, [], state)
        resolved_query = resolve_ambiguous_user_query(raw, state, follow_up)
        canonical = build_resolved_retrieval_query(
            raw_text=raw,
            inference_text=resolved_query.resolved,
            follow_up=follow_up,
            discourse=state,
            history=[],
            resolved_query=resolved_query,
        )
        telemetry = canonical.to_telemetry_dict()
        self.assertEqual(telemetry["resolved_query_raw"], raw)
        self.assertIn("Kathmandu", telemetry["resolved_query_web"])
        self.assertEqual(telemetry["inference_rewrite_reason"], resolved_query.rewrite_reason)


if __name__ == "__main__":
    unittest.main()
