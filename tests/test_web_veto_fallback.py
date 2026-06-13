"""Tests for web-veto fallback prompt wiring."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.memory_filters import (
    EXPLICIT_WEB_EMPTY_SUFFIX,
    PREFERENCE_APPLICATION_SUFFIX,
    WEB_CAPABILITY_DISABLED_SUFFIX,
    detect_explicit_web_request,
    query_implies_live_web_intent,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt


class WebVetoFallbackTests(unittest.TestCase):
    def test_web_capability_blocked_suffix(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            web_capability_blocked=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(WEB_CAPABILITY_DISABLED_SUFFIX.strip()[:40], system)

    def test_joke_query_without_live_web_intent(self):
        self.assertFalse(query_implies_live_web_intent("Tell me a joke."))

    def test_weather_query_implies_live_web_intent(self):
        self.assertTrue(
            query_implies_live_web_intent("What's the weather in Copenhagen today?")
        )

    def test_embedding_only_web_score_does_not_imply_live_intent(self):
        self.assertFalse(
            query_implies_live_web_intent(
                "Tell me a joke.",
                decision={"web_score_source": "embedding", "web_score_final": 0.42},
            )
        )

    def test_substring_web_score_requires_non_temporal_tokens(self):
        self.assertFalse(
            query_implies_live_web_intent(
                "Tell me a joke.",
                decision={"web_score_source": "substring", "web_score_final": 0.2},
            )
        )
        self.assertTrue(
            query_implies_live_web_intent(
                "Find recent news about the Federal Reserve.",
                decision={"web_score_source": "substring", "web_score_final": 0.077},
            )
        )

    def test_hybrid_skips_internet_without_live_web_intent(self):
        from core.memory_filters import should_run_internet_search_for_route

        self.assertFalse(
            should_run_internet_search_for_route(
                "HYBRID",
                "Tell me a joke.",
                decision={"web_score_source": "embedding", "web_score_final": 0.63},
            )
        )

    def test_hybrid_runs_internet_for_weather_query(self):
        from core.memory_filters import should_run_internet_search_for_route

        self.assertTrue(
            should_run_internet_search_for_route(
                "HYBRID",
                "What's the weather in Copenhagen today?",
            )
        )

    def test_web_route_always_searches(self):
        from core.memory_filters import should_run_internet_search_for_route

        self.assertTrue(
            should_run_internet_search_for_route("WEB", "hello")
        )

    def test_detect_explicit_web_request_look_online(self):
        self.assertTrue(detect_explicit_web_request("Look online for a joke."))

    def test_detect_explicit_web_request_search_online(self):
        self.assertTrue(detect_explicit_web_request("Can you search online for a joke?"))

    def test_detect_explicit_web_request_online_for_any_topic(self):
        self.assertTrue(detect_explicit_web_request("Look online for a recipe."))
        self.assertTrue(detect_explicit_web_request("Find reviews online for this phone."))

    def test_plain_chat_not_explicit_web(self):
        self.assertFalse(detect_explicit_web_request("Tell me a story."))

    def test_explicit_web_empty_suffix(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            explicit_web_empty_results=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(EXPLICIT_WEB_EMPTY_SUFFIX.strip()[:40], system)
        self.assertIn("[W]", system)

    def test_disabled_web_suffix_forbids_w_citation(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            web_capability_blocked=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn("[W]", system)
        self.assertIn("Do NOT emit bracket citation tokens", system)

    def test_preference_suffix_when_requested(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            apply_preference_suffix=True,
            preference_context="User prefs: metric units. Apply silently.",
        )
        system = compose_system_prompt(blocks)
        self.assertIn(PREFERENCE_APPLICATION_SUFFIX.strip()[:30], system)
        self.assertIn("metric units", system)

    def test_explicit_remember_skips_preference_suffix(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=True,
            explicit_remember_body="remember I like tea",
            apply_preference_suffix=True,
            preference_context="User prefs: metric units.",
        )
        system = compose_system_prompt(blocks)
        self.assertNotIn(PREFERENCE_APPLICATION_SUFFIX.strip()[:30], system)


if __name__ == "__main__":
    unittest.main()
