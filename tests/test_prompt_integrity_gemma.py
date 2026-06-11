"""Gemma turn-template prompt integrity checks."""
from __future__ import annotations

import unittest

from core.prompt_integrity_validator import validate_chat_inference


class TestPromptIntegrityGemma(unittest.TestCase):
    def test_gemma_multi_turn_prompt_has_assistant_anchor(self) -> None:
        prompt = (
            "<bos><|turn>system\nYou are Qube.<turn|>\n"
            "<|turn>user\nWhat is Nepal known for?<turn|>\n"
            "<|turn>model\nMusic and arts.<turn|>\n"
            "<|turn>user\nTell me more.<turn|>\n"
            "<|turn>model\n"
        )
        pv = validate_chat_inference(
            rendered_prompt=prompt,
            messages=[],
            chat_format="chat_template.default",
            merged_stop_tokens=["<turn|>", "</s>"],
            eos_token_str="<turn|>",
            model_metadata={"tokenizer.chat_template": "<|turn>"},
            reconstruction_ok=True,
        )
        self.assertTrue(pv.assistant_anchor_present)
        self.assertEqual(pv.verdict, "OK")


if __name__ == "__main__":
    unittest.main()
