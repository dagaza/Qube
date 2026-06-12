"""Assistive query expansion guards."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import FollowUpClassification, FollowUpKind  # noqa: E402
from core.discourse_state import (  # noqa: E402
    DiscourseState,
    promote_referent_after_assistant,
    update_discourse_state,
)
from core.sidecar_query_rewrite import (  # noqa: E402
    expansion_adds_unanchored_proper_nouns,
    propose_query_expansion,
    resolve_sidecar_discourse_context,
)
from core.sidecar_types import SidecarResult, SidecarTask  # noqa: E402


class TestSidecarQueryRewrite(unittest.TestCase):
    def test_unanchored_proper_noun_detected(self) -> None:
        self.assertTrue(
            expansion_adds_unanchored_proper_nouns(
                "How to beat the Time Eater boss",
                "How to beat this boss",
                "Slay the Spire",
            )
        )

    def test_anchored_expansion_ok(self) -> None:
        self.assertFalse(
            expansion_adds_unanchored_proper_nouns(
                "Tips for Slay the Spire beginners",
                "tips for this",
                "Slay the Spire",
            )
        )

    def test_low_confidence_returns_none(self) -> None:
        client = mock.Mock()
        client.available = True
        client.complete.return_value = SidecarResult(
            ok=True,
            parsed={
                "expanded_query": "Regarding Slay the Spire: tips",
                "confidence": 0.2,
                "topic_source": "discourse_state",
            },
            confidence=0.2,
            task=SidecarTask.query_rewrite,
        )
        follow = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.8)
        state = DiscourseState(active_topic="Slay the Spire", confidence=0.8)
        with mock.patch(
            "core.sidecar_query_rewrite.get_sidecar_query_rewrite_enabled",
            return_value=True,
        ), mock.patch(
            "core.sidecar_query_rewrite.get_sidecar_min_rewrite_confidence",
            return_value=0.6,
        ):
            out = propose_query_expansion(
                "tips for this",
                follow,
                state,
                [],
                client,
            )
        self.assertIsNone(out)

    def test_resolve_context_uses_entity_not_aspect_topic(self) -> None:
        state = DiscourseState(
            active_referent="Kathmandu",
            referent_type="city",
            referent_source="user_question",
            referent_confidence=0.85,
            active_aspect="flora and fauna",
            active_topic="flora and fauna",
            confidence=0.85,
        )
        entity, aspect = resolve_sidecar_discourse_context(state)
        self.assertEqual(entity, "Kathmandu")
        self.assertEqual(aspect, "flora and fauna")

    def test_kathmandu_music_follow_up_passes_entity_to_sidecar(self) -> None:
        flora_user = "What about Kathmandu's flora and fauna?"
        flora_asst = (
            "Kathmandu is dominated by urban vegetation, featuring Jasmine and Marigold."
        )
        music_user = "Ok, how about its music and arts scene?"

        prior = update_discourse_state(
            [{"role": "user", "content": "What is the capital of Nepal?"}],
            None,
            "What is the capital of Nepal?",
        )
        prior = promote_referent_after_assistant(
            user_prompt="What is the capital of Nepal?",
            assistant_text="Kathmandu is the capital of Nepal.",
            prior=prior,
        )
        prior = update_discourse_state(
            [
                {"role": "user", "content": "What is the capital of Nepal?"},
                {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
                {"role": "user", "content": flora_user},
            ],
            prior,
            flora_user,
        )
        prior = promote_referent_after_assistant(
            user_prompt=flora_user,
            assistant_text=flora_asst,
            prior=prior,
        )
        history = [
            {"role": "user", "content": flora_user},
            {"role": "assistant", "content": flora_asst},
            {"role": "user", "content": music_user},
        ]
        state = update_discourse_state(history, prior, music_user)
        follow = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.72)

        client = mock.Mock()
        client.available = True
        client.complete.return_value = SidecarResult(
            ok=True,
            parsed={
                "expanded_query": "Regarding Kathmandu: music and arts scene",
                "confidence": 0.88,
                "topic_source": "discourse_state",
            },
            confidence=0.88,
            task=SidecarTask.query_rewrite,
        )
        with mock.patch(
            "core.sidecar_query_rewrite.get_sidecar_query_rewrite_enabled",
            return_value=True,
        ), mock.patch(
            "core.sidecar_query_rewrite.get_sidecar_min_rewrite_confidence",
            return_value=0.6,
        ):
            out = propose_query_expansion(
                music_user,
                follow,
                state,
                history,
                client,
                tentative_route="hybrid",
                retrieval_query=music_user,
            )

        self.assertIsNotNone(out)
        self.assertIn("Kathmandu", out.expanded_query)
        call_kwargs = client.complete.call_args.kwargs
        self.assertEqual(call_kwargs.get("topic"), "Kathmandu")
        self.assertIn("music", (call_kwargs.get("active_aspect") or "").lower())
        self.assertEqual(call_kwargs.get("tentative_route"), "hybrid")
        self.assertIn("music", (call_kwargs.get("retrieval_query") or "").lower())

    def test_recommended_target_parsed_telemetry_only(self) -> None:
        client = mock.Mock()
        client.available = True
        client.complete.return_value = SidecarResult(
            ok=True,
            parsed={
                "expanded_query": "Regarding Slay the Spire: tips",
                "confidence": 0.88,
                "topic_source": "discourse_state",
                "recommended_target": "rag",
            },
            confidence=0.88,
            task=SidecarTask.query_rewrite,
        )
        follow = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.8)
        state = DiscourseState(active_topic="Slay the Spire", confidence=0.8)
        with mock.patch(
            "core.sidecar_query_rewrite.get_sidecar_query_rewrite_enabled",
            return_value=True,
        ), mock.patch(
            "core.sidecar_query_rewrite.get_sidecar_min_rewrite_confidence",
            return_value=0.6,
        ):
            out = propose_query_expansion(
                "tips for this",
                follow,
                state,
                [],
                client,
                tentative_route="rag",
                retrieval_query="tips for this",
            )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.recommended_target, "rag")


if __name__ == "__main__":
    unittest.main()
