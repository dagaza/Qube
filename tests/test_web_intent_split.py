"""Tests for hard-web vs live-data intent separation."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.memory_filters import (
    detect_explicit_web_request,
    detect_hard_explicit_web_request,
    query_implies_live_web_intent,
)
from core.router_evaluation import RouterEvalConfig, simulate_execution_route
from mcp.cognitive_router import CognitiveRouterV4


def _simulate(prompt: str, *, internet_enabled: bool = True) -> str:
    router = CognitiveRouterV4()
    decision = router.route(prompt, intent_vector=None)
    route, _ = simulate_execution_route(
        prompt=prompt,
        decision=decision,
        config=RouterEvalConfig(
            internet_enabled=internet_enabled,
            internet_hybrid_auto=True,
            install_centroids=False,
        ),
    )
    return route.upper()


class HardWebDetectionTests(unittest.TestCase):
    def test_search_online_is_hard_web(self) -> None:
        q = "Search online for flight delays today"
        self.assertTrue(detect_hard_explicit_web_request(q))
        self.assertTrue(detect_explicit_web_request(q))

    def test_google_is_hard_web(self) -> None:
        q = "Google the latest NVIDIA earnings"
        self.assertTrue(detect_hard_explicit_web_request(q))

    def test_look_on_web_is_hard_web(self) -> None:
        q = "Look on the web for reviews"
        self.assertTrue(detect_hard_explicit_web_request(q))

    def test_schedule_today_is_not_hard_web(self) -> None:
        for q in (
            "Please remind me what was on my schedule today",
            "Remind me about my schedule for today",
        ):
            self.assertFalse(
                detect_hard_explicit_web_request(q),
                msg=q,
            )


class LiveDataIntentTests(unittest.TestCase):
    def test_weather_queries_imply_live_web(self) -> None:
        for q in (
            "What's the weather today?",
            "Today's weather",
            "Weather in Copenhagen today",
            "Latest weather forecast",
        ):
            self.assertTrue(query_implies_live_web_intent(q), msg=q)

    def test_live_data_survivors(self) -> None:
        for q in (
            "Search for live score of today's Champions League match",
            "Look up the current USD to EUR exchange rate",
            "Current air quality in Delhi today?",
        ):
            self.assertTrue(query_implies_live_web_intent(q), msg=q)

    def test_adversarial_narratives_do_not_imply_live_web(self) -> None:
        for q in (
            "The weather today was lovely on our hike.",
            "I read the news today oh boy.",
        ):
            self.assertFalse(query_implies_live_web_intent(q), msg=q)
            self.assertFalse(detect_hard_explicit_web_request(q), msg=q)

    def test_bare_temporal_substring_does_not_imply_live_web(self) -> None:
        decision = {"web_score_source": "substring", "web_score_final": 0.154}
        self.assertFalse(
            query_implies_live_web_intent(
                "Please remind me what was on my schedule today",
                decision=decision,
            )
        )

    def test_bare_temporal_personal_queries_do_not_imply_live_web(self) -> None:
        for q in (
            "What is my current project status",
            "What am I working on right now",
        ):
            self.assertFalse(query_implies_live_web_intent(q), msg=q)
            self.assertFalse(detect_hard_explicit_web_request(q), msg=q)

    def test_non_temporal_substring_still_implies_live_web(self) -> None:
        decision = {"web_score_source": "substring", "web_score_final": 0.077}
        self.assertTrue(
            query_implies_live_web_intent(
                "Find recent news about the Federal Reserve.",
                decision=decision,
            )
        )


class SimulateExecutionRouteSplitTests(unittest.TestCase):
    def test_schedule_queries_stay_off_web(self) -> None:
        self.assertEqual(
            _simulate("Please remind me what was on my schedule today"),
            "NONE",
        )
        # Recall fusion upgrades to HYBRID (local memory + docs), not WEB.
        self.assertEqual(
            _simulate("Remind me about my schedule for today"),
            "HYBRID",
        )

    def test_weather_queries_route_web(self) -> None:
        for q in (
            "What's the weather today?",
            "Today's weather",
            "Weather in Copenhagen today",
            "Latest weather forecast",
        ):
            self.assertEqual(_simulate(q), "WEB", msg=q)

    def test_hard_commands_force_web(self) -> None:
        for q in (
            "Search online for flight delays today",
            "Google the latest NVIDIA earnings",
            "Look on the web for reviews",
        ):
            self.assertEqual(_simulate(q), "WEB", msg=q)
            self.assertTrue(detect_hard_explicit_web_request(q))

    def test_adversarial_corpus_stays_off_web(self) -> None:
        for q in (
            "The weather today was lovely on our hike.",
            "I read the news today oh boy.",
        ):
            self.assertEqual(_simulate(q), "NONE", msg=q)

    def test_live_data_survivors_route_web(self) -> None:
        for q in (
            "Search for live score of today's Champions League match",
            "Look up the current USD to EUR exchange rate",
            "Current air quality in Delhi today?",
        ):
            self.assertEqual(_simulate(q), "WEB", msg=q)


if __name__ == "__main__":
    unittest.main()
