"""Assistive query expansion guards."""
from __future__ import annotations

import unittest
from unittest import mock

from core.discourse_intent import FollowUpClassification, FollowUpKind
from core.discourse_state import DiscourseState
from core.sidecar_query_rewrite import (
    expansion_adds_unanchored_proper_nouns,
    propose_query_expansion,
)
from core.sidecar_types import SidecarResult, SidecarTask


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


if __name__ == "__main__":
    unittest.main()
