from __future__ import annotations

import unittest

from core.adaptive_retry import (
    maybe_retry,
    skip_retry_for_medium_degeneration,
    skip_retry_for_structured_enumeration_degeneration,
)
from core.output_validation import OutputValidationResult
from core.prompt_contract import PromptContract
from core.reply_shape_policy import ReplyShapePolicy


class _FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[PromptContract] = []

    def execute_from_contract(self, contract: PromptContract, messages: list[dict]) -> str:
        _ = messages
        self.calls.append(contract)
        if self.outputs:
            return self.outputs.pop(0)
        return ""


def _contract(template_source: str, chat_format: str = "chatml") -> PromptContract:
    return PromptContract(
        mode="messages",
        chat_format=chat_format,
        prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        stop=[],
        template_source=template_source,  # type: ignore[arg-type]
        confidence="medium",
    )


def _enumeration_model(model: _FakeModel) -> _FakeModel:
    model._last_reply_shape_policy = ReplyShapePolicy(
        chat_format_mode="structured",
        format_intent="enumeration",
        allow_structured_output=True,
        require_list_format=True,
        system_reply_hint="",
        instruction_conflicts=(),
        resolution_notes=(),
    )
    return model


class TestAdaptiveRetry(unittest.TestCase):
    def test_gguf_failure_retries_to_chatml(self) -> None:
        model = _FakeModel(outputs=["Safe final answer."])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["template_leakage"], severity="high")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "[INST] bad",
            v,
        )
        self.assertTrue(outcome.retry_attempted)
        self.assertTrue(outcome.retry_used)
        self.assertEqual(outcome.text, "Safe final answer.")
        self.assertEqual(outcome.contract.chat_format, "chatml")
        self.assertEqual(len(model.calls), 1)

    def test_chatml_failure_retries_to_rendered(self) -> None:
        model = _FakeModel(outputs=["Safe answer from rendered retry."])
        c = _contract("override", "chatml")
        v = OutputValidationResult(is_valid=False, issues=["role_confusion"], severity="high")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "User: hi",
            v,
        )
        self.assertTrue(outcome.retry_used)
        self.assertEqual(outcome.text, "Safe answer from rendered retry.")
        self.assertEqual(outcome.contract.mode, "rendered")
        self.assertIsNone(outcome.contract.messages)
        self.assertTrue((outcome.contract.prompt or "").startswith("### Instruction:"))

    def test_no_retry_on_valid_output(self) -> None:
        model = _FakeModel(outputs=["unused"])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=True, issues=[], severity="low")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "Valid output",
            v,
        )
        self.assertFalse(outcome.retry_attempted)
        self.assertFalse(outcome.retry_used)
        self.assertEqual(outcome.text, "Valid output")
        self.assertEqual(outcome.contract, c)
        self.assertEqual(model.calls, [])

    def test_no_retry_on_low_severity(self) -> None:
        model = _FakeModel(outputs=["unused"])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["minor"], severity="low")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "maybe fine",
            v,
        )
        self.assertFalse(outcome.retry_attempted)
        self.assertEqual(outcome.text, "maybe fine")
        self.assertEqual(outcome.contract, c)

    def test_gguf_truncated_only_does_not_retry_to_chatml(self) -> None:
        model = _FakeModel(outputs=["<|channel> bad"])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["truncated_output"], severity="medium")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": 'Say exactly "Hello"'}],
            c,
            "Hello",
            v,
        )
        self.assertFalse(outcome.retry_attempted)
        self.assertEqual(outcome.text, "Hello")
        self.assertEqual(outcome.contract, c)
        self.assertEqual(model.calls, [])

    def test_fallback_template_leakage_retries_to_rendered(self) -> None:
        model = _FakeModel(outputs=["Safe answer from rendered retry."])
        c = _contract("fallback", "chatml")
        v = OutputValidationResult(is_valid=False, issues=["template_leakage"], severity="high")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "<|channel|> bad",
            v,
        )
        self.assertTrue(outcome.retry_used)
        self.assertEqual(outcome.text, "Safe answer from rendered retry.")
        self.assertEqual(outcome.contract.mode, "rendered")
        self.assertEqual(len(model.calls), 1)

    def test_skips_retry_for_enumeration_medium_degeneration(self) -> None:
        model = _enumeration_model(_FakeModel(outputs=["Should not run."]))
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["degeneration"], severity="medium")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "List all cities"}],
            c,
            "- **Alpha** — first city\n- **Beta** — second city",
            v,
        )
        self.assertFalse(outcome.retry_attempted)
        self.assertFalse(outcome.retry_used)
        self.assertEqual(
            outcome.retry_reason, "structured_enumeration_medium_degeneration"
        )
        self.assertEqual(model.calls, [])

    def test_still_retries_high_severity_on_enumeration_turn(self) -> None:
        model = _enumeration_model(_FakeModel(outputs=["Safe final answer."]))
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["template_leakage"], severity="high")
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "List all cities"}],
            c,
            "[INST] leaked",
            v,
        )
        self.assertTrue(outcome.retry_used)
        self.assertEqual(len(model.calls), 1)

    def test_skips_retry_for_medium_degeneration_only(self) -> None:
        model = _FakeModel(outputs=["Should not run."])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(
            is_valid=False,
            issues=["degeneration"],
            severity="medium",
            degeneration_retry_eligible=False,
        )
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Write an essay"}],
            c,
            "Some prose with mild repetition.",
            v,
        )
        self.assertFalse(outcome.retry_attempted)
        self.assertEqual(outcome.retry_reason, "medium_degeneration_no_retry")
        self.assertEqual(model.calls, [])

    def test_retries_high_confidence_degeneration(self) -> None:
        model = _FakeModel(outputs=["Safe final answer."])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(
            is_valid=False,
            issues=["degeneration"],
            severity="high",
            degeneration_retry_eligible=True,
            degeneration_score=0.95,
        )
        outcome = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "loop loop loop loop loop loop",
            v,
        )
        self.assertTrue(outcome.retry_attempted)
        self.assertTrue(outcome.retry_used)
        self.assertEqual(len(model.calls), 1)


class TestStructuredEnumerationSkipHelper(unittest.TestCase):
    def test_skip_reason_only_for_medium_degeneration(self) -> None:
        v = OutputValidationResult(is_valid=False, issues=["degeneration"], severity="medium")
        reason = skip_retry_for_structured_enumeration_degeneration(
            v, format_intent="enumeration", require_list_format=True
        )
        self.assertEqual(reason, "structured_enumeration_medium_degeneration")

        high = OutputValidationResult(is_valid=False, issues=["degeneration"], severity="high")
        self.assertIsNone(
            skip_retry_for_structured_enumeration_degeneration(
                high, format_intent="enumeration", require_list_format=True
            )
        )


class TestMediumDegenerationSkipHelper(unittest.TestCase):
    def test_medium_degeneration_skip_reason(self) -> None:
        v = OutputValidationResult(
            is_valid=False,
            issues=["degeneration"],
            severity="medium",
            degeneration_retry_eligible=False,
        )
        self.assertEqual(
            skip_retry_for_medium_degeneration(v),
            "medium_degeneration_no_retry",
        )

    def test_high_confidence_degeneration_not_skipped(self) -> None:
        v = OutputValidationResult(
            is_valid=False,
            issues=["degeneration"],
            severity="high",
            degeneration_retry_eligible=True,
        )
        self.assertIsNone(skip_retry_for_medium_degeneration(v))


if __name__ == "__main__":
    unittest.main()
