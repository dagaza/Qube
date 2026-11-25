"""Search venues ("on Wikipedia", @wikipedia) must not become the discourse topic."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import classify_follow_up  # noqa: E402
from core.discourse_patterns import (  # noqa: E402
    is_source_name,
    mentions_source_venue,
    strip_source_venue_phrases,
)
from core.discourse_query import (  # noqa: E402
    build_resolved_retrieval_query,
    is_deictic_meta_web_request,
    prior_substantive_user_query,
)
from core.discourse_query_rewrite import resolve_ambiguous_user_query  # noqa: E402
from core.discourse_state import update_discourse_state  # noqa: E402

_PRIOR_QUESTION = "What is space made of?"
_PRIOR_ANSWER = (
    "Space is mostly made up of a near-vacuum containing sparse gas, mostly "
    "hydrogen and helium, plus dust, radiation, dark matter and dark energy."
)


def _resolve(prompt: str, *, web_tool_attached: bool = False):
    """Mirror the worker's discourse pipeline for one turn after the space Q&A."""
    history = [
        {"role": "user", "content": _PRIOR_QUESTION},
        {"role": "assistant", "content": _PRIOR_ANSWER},
        {"role": "user", "content": prompt},
    ]
    state = update_discourse_state(history, None, prompt)
    follow_up = classify_follow_up(prompt, history, state)
    resolved_query = resolve_ambiguous_user_query(prompt, state, follow_up)
    inference = resolved_query.resolved if resolved_query.succeeded else prompt
    resolved = build_resolved_retrieval_query(
        raw_text=prompt,
        inference_text=inference,
        follow_up=follow_up,
        discourse=state,
        history=history,
        resolved_query=resolved_query,
        web_tool_attached=web_tool_attached,
    )
    return state, resolved


class TestSourceVenuePatterns(unittest.TestCase):
    def test_source_names(self) -> None:
        for name in ("Wikipedia", "the web", "the internet", "PubMed", "Wikipedia please"):
            self.assertTrue(is_source_name(name), name)
        for topic in ("dark matter", "Google Scholar citations", "Space"):
            self.assertFalse(is_source_name(topic), topic)

    def test_strip_venue_phrases(self) -> None:
        self.assertEqual(
            strip_source_venue_phrases("can you check that on Wikipedia?"),
            "can you check that?",
        )
        self.assertEqual(
            strip_source_venue_phrases("@wikipedia look it up online please"),
            "look it up please",
        )
        self.assertEqual(
            strip_source_venue_phrases("@[tool:wikipedia] verify this via the web"),
            "verify this",
        )

    def test_company_names_are_not_venues_with_ordinary_prepositions(self) -> None:
        self.assertFalse(mentions_source_venue("Who is the CEO at Google?"))
        self.assertFalse(mentions_source_venue("What is wrong with Bing?"))
        self.assertTrue(mentions_source_venue("search for it on Google"))


class TestDeicticMetaWebRequest(unittest.TestCase):
    def test_verify_phrasings_with_venue(self) -> None:
        for prompt in (
            "can you check that on Wikipedia?",
            "verify this on the web",
            "fact-check that online",
            "double check it on PubMed",
            "look that up on Wikipedia",
        ):
            self.assertTrue(is_deictic_meta_web_request(prompt), prompt)

    def test_attached_web_tool_counts_as_explicit_web(self) -> None:
        self.assertFalse(is_deictic_meta_web_request("is that true?"))
        self.assertTrue(is_deictic_meta_web_request("is that true?", web_tool_attached=True))
        self.assertTrue(is_deictic_meta_web_request("@wikipedia can you check that?"))
        self.assertTrue(is_deictic_meta_web_request("@[tool:pubmed] confirm that"))

    def test_non_web_tool_token_is_not_explicit_web(self) -> None:
        self.assertFalse(is_deictic_meta_web_request("@summarize check that"))

    def test_own_subject_is_not_meta(self) -> None:
        for prompt in (
            "what about dark matter?",
            "check that dark matter exists",
            "verify that the sun is a star on Wikipedia",
        ):
            self.assertFalse(
                is_deictic_meta_web_request(prompt, web_tool_attached=True), prompt
            )

    def test_meta_turn_in_history_is_skipped_for_prior_query(self) -> None:
        history = [
            {"role": "user", "content": _PRIOR_QUESTION},
            {"role": "assistant", "content": _PRIOR_ANSWER},
            {"role": "user", "content": "@wikipedia can you check that on Wikipedia?"},
            {"role": "assistant", "content": "Wikipedia says ..."},
        ]
        self.assertEqual(
            prior_substantive_user_query(history, "is that right?"),
            _PRIOR_QUESTION,
        )


class TestWikipediaFollowUpRegression(unittest.TestCase):
    def test_check_that_on_wikipedia_searches_prior_question(self) -> None:
        state, resolved = _resolve("can you check that on Wikipedia?", web_tool_attached=True)
        self.assertNotEqual((state.active_topic or "").lower(), "wikipedia")
        self.assertNotEqual((state.active_referent or "").lower(), "wikipedia")
        self.assertEqual(resolved.web_text, _PRIOR_QUESTION)
        self.assertEqual(resolved.web_rewrite_reason, "meta_prior_turn")

    def test_look_that_up_on_wikipedia_searches_prior_question(self) -> None:
        state, resolved = _resolve("look that up on Wikipedia", web_tool_attached=True)
        self.assertNotEqual((state.active_topic or "").lower(), "wikipedia")
        self.assertEqual(resolved.web_text, _PRIOR_QUESTION)

    def test_venue_phrase_alone_triggers_prior_turn_without_tool(self) -> None:
        _, resolved = _resolve("can you check that on Wikipedia?")
        self.assertEqual(resolved.web_text, _PRIOR_QUESTION)

    def test_plain_check_that_keeps_topic_expansion(self) -> None:
        for prompt in ("can you check that?", "can you verify that?"):
            state, resolved = _resolve(prompt)
            self.assertEqual(state.active_topic, "Space", prompt)
            # S6: the web anchor is the subject as the user wrote it ("What is
            # space made of?"); search is case-insensitive.
            self.assertIn("space", resolved.web_text.lower(), prompt)
            self.assertNotEqual(resolved.web_rewrite_reason, "meta_prior_turn", prompt)

    def test_tool_with_new_subject_searches_new_subject(self) -> None:
        state, resolved = _resolve("what about dark matter?", web_tool_attached=True)
        self.assertEqual(state.active_topic, "dark matter")
        self.assertEqual(resolved.web_text, "what about dark matter?")
        self.assertEqual(resolved.web_rewrite_reason, "none")

    def test_check_that_clause_is_not_rewritten(self) -> None:
        for prompt in (
            "check that dark matter exists on Wikipedia",
            "check that dark matter exists",
        ):
            state, resolved = _resolve(prompt, web_tool_attached=True)
            self.assertEqual(state.active_topic, "dark matter exists", prompt)
            self.assertEqual(resolved.web_text, prompt, prompt)
            self.assertEqual(resolved.web_rewrite_reason, "none", prompt)


if __name__ == "__main__":
    unittest.main()
