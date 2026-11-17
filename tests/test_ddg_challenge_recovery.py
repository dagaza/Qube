"""User-assisted DuckDuckGo challenge recovery (verify in browser, then resume)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.composer_attachments import resolve_attachment_routing
from core.composer_skills import parse_composer_input
from core.knowledge.discovery import challenge_recovery as cr
from core.knowledge.discovery.backoff import (
    is_provider_in_backoff,
    mark_provider_backoff,
    reset_discovery_backoff,
)
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind, with_discovery_fallback
from core.notification_types import (
    ACTION_DDG_RESUME_SEARCH,
    ACTION_DDG_VERIFY_CHALLENGE,
    ddg_backoff_event,
    ddg_resume_ready_event,
)


def _ddg_outcome(*, sentinel: str | None, signals: tuple[str, ...]) -> SearchOutcome:
    return SearchOutcome(
        kind=SearchOutcomeKind.BOT_CHALLENGE,
        provider="duckduckgo",
        bot_challenge_signals=signals,
        failure_sentinel_reason=sentinel,
    )


class TestOutcomeDetection(unittest.TestCase):
    def test_live_challenge_detected(self) -> None:
        outcome = _ddg_outcome(sentinel="ddg_bot_challenge", signals=("http_202",))
        self.assertTrue(cr.outcome_lost_ddg_to_challenge(outcome))

    def test_backoff_skip_detected(self) -> None:
        outcome = _ddg_outcome(sentinel="ddg_backoff", signals=("backoff_active",))
        self.assertTrue(cr.outcome_lost_ddg_to_challenge(outcome))

    def test_challenge_hidden_behind_wikipedia_fallback_detected(self) -> None:
        primary = _ddg_outcome(sentinel="ddg_bot_challenge", signals=("http_202",))
        fallback = SearchOutcome(
            kind=SearchOutcomeKind.SERP_SUCCESS,
            provider="wikipedia",
            candidate_count=5,
        )
        merged = with_discovery_fallback(
            fallback,
            fallback_from="duckduckgo",
            fallback_reason="bot_challenge",
            primary_outcome=primary,
        )
        self.assertTrue(cr.outcome_lost_ddg_to_challenge(merged))

    def test_budget_and_pacing_are_not_challenges(self) -> None:
        for sentinel, signal in (
            ("burst_budget_exhausted", "burst_budget_exhausted"),
            ("session_budget_exhausted", "session_budget_exhausted"),
            ("ddg_pacing_timeout", "pacing_timeout"),
        ):
            outcome = _ddg_outcome(sentinel=sentinel, signals=(signal,))
            self.assertFalse(cr.outcome_lost_ddg_to_challenge(outcome), sentinel)

    def test_success_and_none_not_detected(self) -> None:
        self.assertFalse(cr.outcome_lost_ddg_to_challenge(None))
        self.assertFalse(
            cr.outcome_lost_ddg_to_challenge(
                SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, candidate_count=3)
            )
        )


class TestPendingBlockedSearch(unittest.TestCase):
    def setUp(self) -> None:
        cr.clear_pending_blocked_search()
        reset_discovery_backoff()

    def tearDown(self) -> None:
        cr.clear_pending_blocked_search()
        reset_discovery_backoff()

    def test_latest_blocked_turn_wins_and_take_clears(self) -> None:
        cr.record_blocked_search(session_id="s1", persist_content="first", web_query="q1")
        cr.record_blocked_search(session_id="s2", persist_content="second", web_query="q2")
        pending = cr.take_pending_blocked_search()
        self.assertEqual((pending.session_id, pending.persist_content), ("s2", "second"))
        self.assertIsNone(cr.take_pending_blocked_search())

    def test_expired_pending_is_dropped(self) -> None:
        cr.record_blocked_search(session_id="s1", persist_content="old", web_query="q")
        with patch.object(cr, "PENDING_BLOCKED_SEARCH_TTL_SEC", -1):
            self.assertIsNone(cr.get_pending_blocked_search())

    def test_empty_content_not_recorded(self) -> None:
        cr.record_blocked_search(session_id="s1", persist_content="  ", web_query="q")
        self.assertIsNone(cr.get_pending_blocked_search())

    def test_lift_pause_clears_backoff_and_resets_session(self) -> None:
        mark_provider_backoff("duckduckgo", reason="bot_challenge")
        with patch(
            "core.knowledge.discovery.ddg_http.reset_ddg_browser_session"
        ) as reset_session:
            cr.lift_ddg_pause_after_user_verification()
        self.assertFalse(is_provider_in_backoff("duckduckgo"))
        reset_session.assert_called_once()


class TestResumePrompt(unittest.TestCase):
    def test_plain_question_gets_internet_pinned(self) -> None:
        raw = cr.resume_prompt_text("What is the LHC?")
        clean, attachments, _skills = parse_composer_input(raw)
        self.assertEqual(clean, "What is the LHC?")
        self.assertEqual([a.id for a in attachments], ["internet"])
        self.assertIsNotNone(resolve_attachment_routing(attachments))

    def test_existing_tool_token_kept(self) -> None:
        raw = "@[tool:recipe] sourdough"
        self.assertEqual(cr.resume_prompt_text(raw), raw)

    def test_verification_url_targets_same_endpoint(self) -> None:
        self.assertEqual(
            cr.ddg_verification_url("lhc uses"),
            "https://html.duckduckgo.com/html/?q=lhc+uses",
        )
        self.assertEqual(cr.ddg_verification_url(None), "https://html.duckduckgo.com/html/")


class TestNotifications(unittest.TestCase):
    def test_backoff_toast_offers_browser_verification(self) -> None:
        event = ddg_backoff_event(remaining_seconds=600)
        self.assertEqual(event.action_id, ACTION_DDG_VERIFY_CHALLENGE)
        self.assertIn("10 more minutes", event.body)

    def test_resume_toast_is_sticky_and_resumes(self) -> None:
        event = ddg_resume_ready_event(has_blocked_turn=True)
        self.assertEqual(event.action_id, ACTION_DDG_RESUME_SEARCH)
        self.assertEqual(event.action_label, "Resume search")
        self.assertEqual(event.auto_dismiss_ms, 0)
        self.assertEqual(
            ddg_resume_ready_event(has_blocked_turn=False).action_label,
            "Re-enable DuckDuckGo",
        )


if __name__ == "__main__":
    unittest.main()
