"""Canonical ResolvedRetrievalQuery builder and web inference alignment."""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import (  # noqa: E402
    FollowUpClassification,
    FollowUpKind,
    classify_follow_up,
)
from core.discourse_query import (  # noqa: E402
    SearchTargetResult,
    build_resolved_retrieval_query,
    resolve_routing_query,
    resolve_search_target,
)
from core.query_resolution_evaluation import build_discourse_resolution  # noqa: E402
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
        # Plan S36 (D53): both start from the inference text; routing keeps the
        # legacy expansion, library and memory search their own resolution.
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
        self.assertEqual(canonical.routing_text, resolve_routing_query(inference, follow_up, state))
        self.assertIn("Kathmandu", canonical.routing_text)
        self.assertIn("Kathmandu", canonical.retrieval_text)

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
        self.assertNotIn("web_query_acronym_expansions", telemetry)


_ANT_HISTORY = [
    {"role": "user", "content": "What is actor-network theory?>"},
    {
        "role": "assistant",
        "content": (
            "Actor-network theory (ANT) is a sociological approach that treats human "
            "and non-human entities as actors. Would you like an example of how ANT "
            "analyzes a specific technology?"
        ),
    },
]
_ANT_TURN = "Yes, please tell me how ANT analyzes Artificial Inteligence for example."
_FIXTURES = os.path.join(_WS_ROOT, "eval", "fixtures", "knowledge_quality")


class TestAcronymExpansion(unittest.TestCase):
    """Plan S5: acronyms defined earlier expand into the web/retrieval query."""

    def test_expansion_does_not_need_a_follow_up_classification(self) -> None:
        # S5: a definition in history is enough, even when the turn is not a
        # follow-up and discourse tracking is off.
        canonical = build_resolved_retrieval_query(
            raw_text=_ANT_TURN,
            inference_text=_ANT_TURN,
            follow_up=FollowUpClassification(FollowUpKind.NONE, 0.0),
            discourse=None,
            history=_ANT_HISTORY,
        )
        self.assertIn("Actor-network theory (ANT)", canonical.web_text)
        self.assertEqual(canonical.web_rewrite_reason, "acronym_expansion")
        self.assertTrue(canonical.web_rewritten)
        self.assertEqual(canonical.acronym_expansions, (("ANT", "Actor-network theory"),))
        telemetry = canonical.to_telemetry_dict()
        self.assertEqual(telemetry["web_query_rewrite_reason"], "acronym_expansion")
        self.assertEqual(telemetry["web_query_acronym_expansions"], ["ANT=Actor-network theory"])

    def test_ant_follow_up_keeps_subject_with_facet(self) -> None:
        # S6: the production failure was kind=NONE / confidence 0.0, and the
        # facet "Artificial Inteligence" replaced the subject.
        follow_up, state, canonical = build_discourse_resolution(_ANT_TURN, _ANT_HISTORY)
        self.assertTrue(follow_up.active)
        self.assertEqual(state.subject, "actor-network theory")
        self.assertEqual(state.facet, "Artificial Inteligence")
        self.assertEqual(state.subject_relation, "continuation")
        self.assertEqual(
            canonical.web_text, "actor-network theory how ANT analyzes Artificial Inteligence"
        )
        self.assertEqual(canonical.web_rewrite_reason, "topic_expansion")
        self.assertNotIn("Regarding", canonical.web_text)
        telemetry = canonical.to_telemetry_dict()
        self.assertEqual(telemetry["web_query_subject"], "actor-network theory")
        self.assertEqual(telemetry["web_query_subject_relation"], "continuation")

    def test_web_and_retrieval_text_are_expanded_routing_is_not(self) -> None:
        # Plan S36 (D53): library and memory search get S5's acronym expansion
        # and S6's subject like the web; routing keeps the legacy text.
        _fu, _state, canonical = build_discourse_resolution(_ANT_TURN, _ANT_HISTORY)
        self.assertEqual(canonical.raw_text, _ANT_TURN)
        self.assertEqual(canonical.retrieval_text, canonical.web_text)
        self.assertEqual(canonical.retrieval_rewrite_reason, "subject")
        self.assertNotIn("Actor-network", canonical.routing_text)
        plain = build_resolved_retrieval_query(
            raw_text=_ANT_TURN,
            inference_text=_ANT_TURN,
            follow_up=FollowUpClassification(FollowUpKind.NONE, 0.0),
            discourse=None,
            history=_ANT_HISTORY,
        )
        self.assertIn("Actor-network theory (ANT)", plain.retrieval_text)
        self.assertNotIn("Actor-network", plain.routing_text)

    def test_topic_switch_is_untouched(self) -> None:
        q = "What is the boiling point of ethanol?"
        _fu, _state, canonical = build_discourse_resolution(q, _ANT_HISTORY)
        self.assertEqual(canonical.web_text, q)
        self.assertEqual(canonical.web_rewrite_reason, "none")
        self.assertEqual(canonical.acronym_expansions, ())

    def test_undefined_acronym_is_untouched(self) -> None:
        q = "What does the WHO say about ANT?"
        _fu, _state, canonical = build_discourse_resolution(q, [])
        self.assertEqual(canonical.web_text, q)
        self.assertEqual(canonical.web_rewrite_reason, "none")

    def test_discourse_rewrite_reason_is_kept_alongside_expansion(self) -> None:
        # The worker keys flags on "topic_expansion" / "meta_prior_turn"; an
        # acronym expansion on top must not hide them.
        # No user turn sets a subject here, so the legacy discourse anchor is
        # used (S6 drops its "Regarding X:" prefix).
        assistant_only = _ANT_HISTORY[1:]
        with patch(
            "core.discourse_query.resolve_search_target",
            return_value=SearchTargetResult("Regarding sociology: how ANT works", "topic_expansion"),
        ):
            _fu, _state, canonical = build_discourse_resolution("how ANT works", assistant_only)
        self.assertEqual(canonical.web_text, "sociology how Actor-network theory (ANT) works")
        self.assertEqual(canonical.web_rewrite_reason, "topic_expansion")
        self.assertEqual(canonical.acronym_expansions, (("ANT", "Actor-network theory"),))

    def test_multi_turn_acronym_fixtures_contain_gold_subject(self) -> None:
        for name in ("mt_ant_followup_live.json", "ant_ai_candidates.json", "mt_acronym_ms_treatment.json"):
            with self.subTest(fixture=name):
                with open(os.path.join(_FIXTURES, name), encoding="utf-8") as fh:
                    fixture = json.load(fh)
                _fu, _state, canonical = build_discourse_resolution(
                    fixture["query_clean"], fixture["history"]
                )
                self.assertIn(fixture["gold_subject"].lower(), canonical.web_text.lower())


if __name__ == "__main__":
    unittest.main()
