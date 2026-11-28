"""Plan S28: routing and the prompt hints read the tracked subject, not ``active_topic`` (D65, D66).

``active_topic`` keeps its legacy value, which can be a facet of the subject ("hotels
there" in a Lisbon trip) or the turn's own words ("my bread last time"). Its readers
now take ``DiscourseState.continued_subject()``: the tracked subject when the turn
continues it, otherwise nothing.

- the router's input text (``routing_text``) anchors on the referent or that subject;
- the follow-up downgrade (MEMORY / RAG / HYBRID -> NONE) fires only on a follow-up
  that continues the subject. With ``active_topic`` it sent S33's memory follow-up
  "What did we decide last time?" and a synthetic switch to NONE, where only
  preferences are recalled (V93);
- the web veto counts a referent or that subject, and ``active_topic`` only where no
  subject was tracked, the one case where the web text still expands with it;
- the prompt grounding ("[Continuing our discussion of X]") and the salience hint.

The follow-up classifier's topic boost, the shadow-policy signal, the pronoun
target and the log lines keep ``active_topic``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.discourse_intent import FollowUpClassification, FollowUpKind, classify_follow_up  # noqa: E402
from core.discourse_prompt_rewrite import (  # noqa: E402
    resolve_discourse_prompt_rewrite,
    select_salience_anchor,
)
from core.discourse_query import should_veto_ungrounded_web_follow_up  # noqa: E402
from core.discourse_query_rewrite import resolve_ambiguous_user_query  # noqa: E402
from core.discourse_state import DiscourseState  # noqa: E402
from core.query_resolution_evaluation import build_discourse_resolution  # noqa: E402
from core.router_evaluation import RouterEvalConfig, simulate_execution_route  # noqa: E402


def _u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


OAUTH = [_u("What is OAuth?"), _a("OAuth is an authorization framework. Would you like an example of the authorization code flow?")]
LEDGER = [_u("What were we working on with the Ledger API?"), _a("The v2 design: the rate-limit section, and before that pagination and idempotency.")]
INSURANCE = [_u("What's the excess on my contents insurance?"), _a("£150 per claim, or £350 for escape of water.")]


class TestContinuedSubject(unittest.TestCase):
    def test_only_a_continuation_has_one(self):
        for relation, expected in (
            ("continuation", "OAuth"),
            ("new_subject", None),
            ("none", None),
            ("no_prior", None),
        ):
            with self.subTest(relation=relation):
                state = DiscourseState(active_topic="the security risks", subject="OAuth", subject_relation=relation)
                self.assertEqual(state.continued_subject(), expected)
        self.assertIsNone(DiscourseState(subject_relation="continuation").continued_subject())

    def test_the_salience_anchor_is_the_referent_then_that_subject(self):
        self.assertEqual(
            DiscourseState(active_referent="Kathmandu", subject="Nepal", subject_relation="continuation").salience_anchor(),
            "Kathmandu",
        )
        self.assertEqual(
            DiscourseState(active_topic="the security risks", subject="OAuth", subject_relation="continuation").salience_anchor(),
            "OAuth",
        )
        self.assertIsNone(DiscourseState(active_topic="my bread last time", subject="my bread last time",
                                         subject_relation="new_subject").salience_anchor())


class TestRoutingText(unittest.TestCase):
    def _routing(self, history, turn):
        _fu, _st, r = build_discourse_resolution(turn, list(history))
        return r.routing_text

    def test_the_subject_anchors_the_router_text(self):
        for history, turn, expected in (
            (OAUTH, "what about the security risks?", "Regarding OAuth: what about the security risks?"),
            ([_u("Where was Mozart born?"), _a("Mozart was born in Salzburg, in 1756.")],
             "What's the best time of year to visit there?",
             "Regarding Salzburg: What's the best time of year to visit there?"),
            ([_u("We discussed using PostgreSQL for the project."), _a("PostgreSQL offers strong relational features.")],
             "Remind me what we decided about that?",
             "Regarding PostgreSQL: Remind me what we decided about that?"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._routing(history, turn), expected)

    def test_a_turn_that_does_not_continue_the_subject_has_no_anchor(self):
        for history, turn in (
            (LEDGER, "What did we decide last time?"),
            (LEDGER, "Remind me what we concluded about my bread last time."),
            (INSURANCE, "How do I reset my wifi router?"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._routing(history, turn), turn)


class TestFollowUpDowngrade(unittest.TestCase):
    """The eval's copy of LLMWorker's downgrade (``simulate_execution_route``)."""

    def _route(self, history, turn, router_route):
        fu, st, _r = build_discourse_resolution(turn, list(history))
        return simulate_execution_route(
            prompt=turn, decision={"route": router_route}, config=RouterEvalConfig(), follow_up=fu, discourse_state=st
        )

    def test_recall_is_not_dropped_for_a_turn_without_the_subject(self):
        # V93, D66: S33's mem_mt_api_decision and the mem_nar_bread switch.
        self.assertEqual(self._route(LEDGER, "What did we decide last time?", "memory"), ("MEMORY", ""))
        self.assertEqual(self._route(LEDGER, "Remind me what we concluded about my bread last time.", "hybrid"), ("HYBRID", ""))

    def test_a_follow_up_on_the_subject_still_stays_in_chat(self):
        self.assertEqual(
            self._route(OAUTH, "what about the security risks?", "rag"), ("NONE", "discourse_follow_up_downgrade")
        )


class TestWebVeto(unittest.TestCase):
    FU = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.72)

    def test_a_legacy_topic_does_not_ground_a_turn_with_its_own_subject(self):
        self.assertTrue(
            should_veto_ungrounded_web_follow_up(
                self.FU, DiscourseState(active_topic="tips for this", subject="tips", subject_relation="new_subject")
            )
        )

    def test_the_continued_subject_grounds_it(self):
        self.assertFalse(
            should_veto_ungrounded_web_follow_up(
                self.FU, DiscourseState(active_topic="tips", subject="Slay the Spire", subject_relation="continuation")
            )
        )

    def test_without_a_tracked_subject_the_legacy_topic_still_grounds_it(self):
        # The web text expands with active_topic only here (``_subject_web_target``).
        self.assertFalse(
            should_veto_ungrounded_web_follow_up(self.FU, DiscourseState(active_topic="Slay the Spire"))
        )


class TestPromptHints(unittest.TestCase):
    def _hints(self, history, turn):
        _fu, st, _r = build_discourse_resolution(turn, list(history))
        fu = classify_follow_up(turn, list(history), st)
        rq = resolve_ambiguous_user_query(turn, st, fu)
        grounded = resolve_discourse_prompt_rewrite(user_message=turn, resolved_query=rq, follow_up=fu, discourse=st)
        salience = select_salience_anchor(discourse=st, user_message=turn, resolved_query=rq)[0]
        return grounded.grounded, salience

    def test_the_subject_not_the_facet(self):
        grounded, salience = self._hints(OAUTH, "what about the security risks?")
        self.assertEqual(grounded.split("\n")[0], "[Continuing our discussion of OAuth]")
        self.assertEqual(salience, "OAuth")

    def test_no_anchor_from_the_turns_own_words(self):
        for history, turn in (
            (INSURANCE, "How do I reset my wifi router?"),
            (LEDGER, "Remind me what we concluded about my bread last time."),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._hints(history, turn), (turn, None))


if __name__ == "__main__":
    unittest.main()
