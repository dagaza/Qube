from __future__ import annotations

import unittest

from core.adaptive_retry import maybe_retry
from core.output_validation import OutputValidationResult
from core.prompt_contract import PromptContract


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


class TestAdaptiveRetry(unittest.TestCase):
    def test_gguf_failure_retries_to_chatml(self) -> None:
        model = _FakeModel(outputs=["Safe final answer."])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["template_leakage"], severity="high")
        out, final_contract, used = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "[INST] bad",
            v,
        )
        self.assertTrue(used)
        self.assertEqual(out, "Safe final answer.")
        self.assertEqual(final_contract.chat_format, "chatml")
        self.assertEqual(len(model.calls), 1)

    def test_chatml_failure_retries_to_rendered(self) -> None:
        model = _FakeModel(outputs=["Safe answer from rendered retry."])
        c = _contract("override", "chatml")
        v = OutputValidationResult(is_valid=False, issues=["role_confusion"], severity="high")
        out, final_contract, used = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "User: hi",
            v,
        )
        self.assertTrue(used)
        self.assertEqual(out, "Safe answer from rendered retry.")
        self.assertEqual(final_contract.mode, "rendered")
        self.assertIsNone(final_contract.messages)
        self.assertTrue((final_contract.prompt or "").startswith("### Instruction:"))

    def test_no_retry_on_valid_output(self) -> None:
        model = _FakeModel(outputs=["unused"])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=True, issues=[], severity="low")
        out, final_contract, used = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "Valid output",
            v,
        )
        self.assertFalse(used)
        self.assertEqual(out, "Valid output")
        self.assertEqual(final_contract, c)
        self.assertEqual(model.calls, [])

    def test_no_retry_on_low_severity(self) -> None:
        model = _FakeModel(outputs=["unused"])
        c = _contract("gguf", "chat_template.default")
        v = OutputValidationResult(is_valid=False, issues=["minor"], severity="low")
        out, final_contract, used = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "maybe fine",
            v,
        )
        self.assertFalse(used)
        self.assertEqual(out, "maybe fine")
        self.assertEqual(final_contract, c)

    def test_fallback_template_leakage_retries_to_rendered(self) -> None:
        model = _FakeModel(outputs=["Safe answer from rendered retry."])
        c = _contract("fallback", "chatml")
        v = OutputValidationResult(is_valid=False, issues=["template_leakage"], severity="high")
        out, final_contract, used = maybe_retry(
            model,
            [{"role": "user", "content": "Hello"}],
            c,
            "<|channel|> bad",
            v,
        )
        self.assertTrue(used)
        self.assertEqual(out, "Safe answer from rendered retry.")
        self.assertEqual(final_contract.mode, "rendered")
        self.assertEqual(len(model.calls), 1)
