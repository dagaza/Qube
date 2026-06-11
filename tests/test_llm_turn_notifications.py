from __future__ import annotations

import unittest

from core.adaptive_retry import maybe_retry
from core.notification_policy import plan_delivery
from core.notification_types import (
    NotificationSeverity,
    format_retry_in_progress_event,
    output_truncated_max_tokens_event,
)
from core.output_validation import OutputValidationResult
from core.prompt_contract import PromptContract
from core.reply_shape_policy import ReplyShapePolicy


class TestTurnNotificationEvents(unittest.TestCase):
    def test_max_tokens_event_is_warning_turn(self) -> None:
        event = output_truncated_max_tokens_event(session_id="sess-1")
        self.assertEqual(event.severity, NotificationSeverity.WARNING)
        self.assertEqual(event.category, "turn")
        self.assertIn("maximum length", event.body.lower())

    def test_format_retry_event_mentions_degeneration(self) -> None:
        event = format_retry_in_progress_event(
            session_id="sess-2",
            issues=["degeneration"],
        )
        self.assertIn("re-generating", event.body.lower())
        self.assertIn("repetition", event.body.lower())

    def test_warning_shows_in_app_when_focused(self) -> None:
        event = output_truncated_max_tokens_event(session_id="sess-3")
        plan = plan_delivery(event, window_visible=True, window_focused=True)
        self.assertTrue(plan.show_in_app)


class TestAdaptiveRetryNoticeHook(unittest.TestCase):
    def test_emits_notice_before_retry_execute(self) -> None:
        notices: list[tuple[str, dict]] = []

        class Model:
            def _turn_notice_hook(self, kind: str, payload: dict | None = None) -> None:
                notices.append((kind, dict(payload or {})))

            def execute_from_contract(self, contract, messages) -> str:
                _ = contract, messages
                return "Safe final answer."

        model = Model()
        contract = PromptContract(
            mode="messages",
            chat_format="chat_template.default",
            prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            stop=[],
            template_source="gguf",
            confidence="medium",
        )
        validation = OutputValidationResult(
            is_valid=False,
            issues=["degeneration"],
            severity="high",
        )
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            contract,
            "loop loop loop loop",
            validation,
        )
        self.assertTrue(outcome.retry_used)
        self.assertEqual(notices, [("format_retry", {"issues": ["degeneration"]})])

    def test_skipped_enumeration_retry_does_not_emit_notice(self) -> None:
        notices: list[tuple[str, dict]] = []

        class Model:
            _last_reply_shape_policy = ReplyShapePolicy(
                chat_format_mode="structured",
                format_intent="enumeration",
                allow_structured_output=True,
                require_list_format=True,
                system_reply_hint="",
                instruction_conflicts=(),
                resolution_notes=(),
            )

            def _turn_notice_hook(self, kind: str, payload: dict | None = None) -> None:
                notices.append((kind, dict(payload or {})))

        model = Model()
        contract = PromptContract(
            mode="messages",
            chat_format="chat_template.default",
            prompt=None,
            messages=[{"role": "user", "content": "List items"}],
            stop=[],
            template_source="gguf",
            confidence="medium",
        )
        validation = OutputValidationResult(
            is_valid=False,
            issues=["degeneration"],
            severity="medium",
        )
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "List items"}],
            contract,
            "- **A** — one",
            validation,
        )
        self.assertFalse(outcome.retry_attempted)
        self.assertEqual(notices, [])


if __name__ == "__main__":
    unittest.main()
