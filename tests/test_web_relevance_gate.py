"""Web retrieval relevance gate and pipeline helpers."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import classify_follow_up  # noqa: E402
from core.discourse_query import (  # noqa: E402
    resolve_search_target,
    resolve_web_query,
    web_query_rewrite_failed,
)
from core.discourse_state import update_discourse_state  # noqa: E402
from core.retrieval_relevance import (  # noqa: E402
    DEFAULT_WEB_MIN_TOKEN_OVERLAP,
    filter_web_results,
    passes_web_relevance_gate,
    query_snippet_token_overlap,
)


class TestQuerySnippetOverlap(unittest.TestCase):
    def test_birds_query_matches_bird_snippet(self) -> None:
        q = "Why do birds take dust baths?"
        snippet = "Birds take dust baths to remove parasites from feathers."
        overlap = query_snippet_token_overlap(q, snippet)
        self.assertGreaterEqual(overlap, DEFAULT_WEB_MIN_TOKEN_OVERLAP)
        self.assertTrue(passes_web_relevance_gate(q, "Dust bathing", snippet))

    def test_meta_query_rejects_unrelated_snippet(self) -> None:
        q = "Can you also do an online search for the answer?"
        snippet = "Learn how to configure online search settings in your browser."
        overlap = query_snippet_token_overlap(q, snippet)
        self.assertLess(overlap, DEFAULT_WEB_MIN_TOKEN_OVERLAP)
        self.assertFalse(passes_web_relevance_gate(q, "Online search", snippet))


class TestFilterWebResults(unittest.TestCase):
    def test_filters_irrelevant_hits(self) -> None:
        query = "Why do birds take dust baths?"
        items = [
            {
                "title": "Online search tips",
                "snippet": "How to search the web for answers to your questions.",
            },
            {
                "title": "Dust bathing in birds",
                "snippet": "Many bird species take dust baths to clean feathers.",
            },
        ]
        kept, diag = filter_web_results(
            query,
            items,
            use_embedding_gate=False,
        )
        self.assertEqual(diag["web_results_raw_count"], 2)
        self.assertEqual(len(kept), 1)
        self.assertIn("bird", kept[0]["snippet"].lower())

    def test_empty_when_all_irrelevant(self) -> None:
        query = "Why do birds take dust baths?"
        items = [
            {"title": "Search online", "snippet": "Use online search engines today."},
            {"title": "Web tips", "snippet": "Configure your browser search bar."},
        ]
        kept, diag = filter_web_results(query, items, use_embedding_gate=False)
        self.assertEqual(len(kept), 0)
        self.assertEqual(diag["web_results_kept_count"], 0)
        self.assertEqual(len(diag["web_relevance_dropped"]), 2)


class TestMetaWebPipeline(unittest.TestCase):
    def test_birds_follow_up_resolves_prior_question(self) -> None:
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
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            update_discourse_state(history, None, history[-1]["content"]),
            history,
        )
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")
        self.assertEqual(target.query, "Why do birds take dust baths?")
        self.assertFalse(
            web_query_rewrite_failed(
                history[-1]["content"],
                follow_up,
                target.query,
                explicit_web=True,
            )
        )

    def test_unresolved_meta_flags_rewrite_failed(self) -> None:
        prompt = "Can you search online for the answer?"
        follow_up = classify_follow_up(prompt, [], None)
        self.assertTrue(
            web_query_rewrite_failed(
                prompt,
                follow_up,
                prompt,
                explicit_web=True,
            )
        )

    def test_resolve_web_query_matches_search_target(self) -> None:
        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {"role": "assistant", "content": "Birds dust-bathe."},
            {"role": "user", "content": "Search online for the answer please."},
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        web_q = resolve_web_query(history[-1]["content"], follow_up, state, history)
        target = resolve_search_target(
            history[-1]["content"], follow_up, state, history
        )
        self.assertEqual(web_q, target.query)

    def test_meta_web_rewrite_skips_tool_only_prior_turn(self) -> None:
        history = [
            {"role": "user", "content": "What is the capital of Romania?"},
            {"role": "assistant", "content": "Bucharest is the capital of Romania."},
            {"role": "user", "content": "And how about its population size?"},
            {"role": "assistant", "content": "About 1.8 million in the city proper."},
            {"role": "user", "content": "@[tool:internet]"},
            {
                "role": "user",
                "content": "Can you also check online for the answer?",
            },
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            state,
            history,
        )
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")
        self.assertIn("Bucharest", target.query)
        self.assertIn("population", target.query.lower())

    def test_meta_web_rewrite_grounds_deictic_prior_with_referent(self) -> None:
        history = [
            {"role": "user", "content": "What is the capital of Romania?"},
            {"role": "assistant", "content": "Bucharest is the capital of Romania."},
            {"role": "user", "content": "Great, and how about its surface area?"},
            {"role": "assistant", "content": "About 228 km²."},
            {
                "role": "user",
                "content": "Can you check for the answer online to that last question?",
            },
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            state,
            history,
        )
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")
        self.assertEqual(target.query, "What is the surface area of Bucharest?")


if __name__ == "__main__":
    unittest.main()
