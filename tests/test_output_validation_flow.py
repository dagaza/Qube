from __future__ import annotations

import unittest

from core.output_validation_flow import run_output_validation_and_retry
from core.output_validation_sanitize import sanitize_output_for_validation
from core.prompt_contract import PromptContract


class _FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.last_max_tokens: int | None = None

    def execute_from_contract(self, contract: PromptContract, messages: list[dict]) -> str:
        _ = contract, messages
        self.last_max_tokens = getattr(self, "_adaptive_retry_max_tokens", None)
        if self.outputs:
            return self.outputs.pop(0)
        return ""


def _contract() -> PromptContract:
    return PromptContract(
        mode="messages",
        chat_format="chat_template.default",
        prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        stop=[],
        template_source="gguf",
        confidence="medium",
    )


class TestOutputValidationFlow(unittest.TestCase):
    def test_sanitized_gemma_output_skips_retry(self) -> None:
        body = "| Aspect | Value |\n| --- | --- |"
        raw = f"<|channel>thought\nPlan.\n\n{body}"
        model = _FakeModel(outputs=["unused"])
        out, _, trace, validation = run_output_validation_and_retry(
            model,
            final_text=raw,
            contract=_contract(),
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=2048,
        )
        self.assertTrue(validation.is_valid)
        self.assertFalse(trace.retry_used)
        self.assertFalse(trace.retry_attempted)
        self.assertIn(body, sanitize_output_for_validation(raw))
        self.assertEqual(out, raw)

    def test_retry_uses_original_token_budget(self) -> None:
        model = _FakeModel(outputs=["Safe final answer."])
        # Survives sanitization — only sanitized text gates retry (not raw [INST] leaks).
        raw = "User: hello\nAssistant: leaked template dialog"
        _, _, trace, _ = run_output_validation_and_retry(
            model,
            final_text=raw,
            contract=_contract(),
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=2048,
        )
        self.assertTrue(trace.retry_used)
        self.assertTrue(trace.retry_attempted)
        self.assertEqual(model.last_max_tokens, 2048)
        self.assertEqual(trace.retry_max_tokens, 2048)


    def test_enumeration_flow_skips_medium_degeneration_retry(self) -> None:
        from core.reply_shape_policy import ReplyShapePolicy

        bullets = _kathmandu_bullets()
        model = _FakeModel(outputs=["Should not run."])
        model._last_reply_shape_policy = ReplyShapePolicy(
            chat_format_mode="structured",
            format_intent="enumeration",
            allow_structured_output=True,
            require_list_format=True,
            system_reply_hint="",
            instruction_conflicts=(),
            resolution_notes=(),
        )
        out, _, trace, validation = run_output_validation_and_retry(
            model,
            final_text=bullets,
            contract=_contract(),
            messages=[{"role": "user", "content": "List everything"}],
            max_tokens=2048,
        )
        self.assertIn("degeneration", validation.issues)
        self.assertFalse(trace.retry_attempted)
        self.assertEqual(trace.retry_reason, "structured_enumeration_medium_degeneration")
        self.assertEqual(out, bullets)


def _kathmandu_bullets() -> str:
    items = [
        ("History", "Ancient city with deep religious significance."),
        ("Culture", "A vibrant confluence of Hinduism and Buddhism."),
        ("Economy", "Mixed economy reliant on tourism and trade."),
        ("Architecture", "Intricately carved wooden structures and pagodas."),
        ("Cuisine", "Rich vegetarian and meat-based dishes."),
        ("Music", "Traditional devotional sounds and contemporary fusion."),
        ("Literature", "Epic poetry, religious texts, and modern prose."),
        ("Visual Arts", "Temple iconography, metalwork, and Thangka painting."),
    ]
    table = "| Feature | Overview |\n| --- | --- |\n" + "\n".join(
        f"| **{title}** | {desc} |" for title, desc in items
    )
    music = "\n".join(
        f"- **Item {i}** — Detailed explanation for topic {i} in Kathmandu."
        for i in range(1, 11)
    )
    return f"{table}\n\n### Music\n\n{music}"


if __name__ == "__main__":
    unittest.main()
