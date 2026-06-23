from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.template_output_profile import resolve_template_output_profile


class TestTemplateOutputProfile(unittest.TestCase):
    def test_nemotron_jinja_gets_delimiter_tier(self) -> None:
        llama = MagicMock()
        llama.metadata = {
            "general.name": "NVIDIA-Nemotron-3-Nano-4B",
            "tokenizer.chat_template": "<|im_start|>assistant\n<|assistant|>\n</|assistant|>",
        }
        llama.chat_format = "chat_template.default"
        llama.model_path = "/models/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"
        profile = resolve_template_output_profile(
            llama,
            model_path=llama.model_path,
            harmony_model_active=False,
            effective_chat_format="chat_template.default",
            supports_thinking_tokens=True,
        )
        self.assertEqual(profile.family, "nemotron")
        self.assertEqual(profile.grammar_tier, "delimiter")
        self.assertTrue(profile.jinja_runtime)
        self.assertIn("<|assistant|>", profile.assistant_open_tokens)
        self.assertIn("</|assistant|>", profile.assistant_close_tokens)

    def test_harmony_active_gets_harmony_channel_tier(self) -> None:
        llama = MagicMock()
        llama.metadata = {"general.name": "openai_gpt-oss-20b"}
        llama.chat_format = "chatml"
        llama.model_path = "/models/gpt-oss-20b.gguf"
        profile = resolve_template_output_profile(
            llama,
            model_path=llama.model_path,
            harmony_model_active=True,
            effective_chat_format="chatml",
        )
        self.assertEqual(profile.family, "harmony")
        self.assertEqual(profile.grammar_tier, "harmony_channel")


if __name__ == "__main__":
    unittest.main()
