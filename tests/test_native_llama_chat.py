from __future__ import annotations

import unittest
from unittest.mock import patch

from core.native_llama_chat import prefer_gguf_jinja_chat_format


class _LlamaStub:
    def __init__(self) -> None:
        self.chat_format = "llama-2"
        self.metadata = {"tokenizer.chat_template": "{{ '<|im_start|>user' }}"}
        self.model_path = "/models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
        self._chat_handlers = {"chatml": object(), "chat_template.default": object()}


class TestPreferGgufJinjaChatFormat(unittest.TestCase):
    def test_nemotron_prefers_embedded_gguf_template_over_chatml(self) -> None:
        llama = _LlamaStub()
        with patch(
            "core.native_llama_chat.get_internal_native_chat_format",
            return_value="auto",
        ):
            prefer_gguf_jinja_chat_format(llama)
        self.assertEqual(llama.chat_format, "chat_template.default")

    def test_nemotron_falls_back_to_chatml_without_embedded_template(self) -> None:
        llama = _LlamaStub()
        llama.metadata = {}
        with patch(
            "core.native_llama_chat.get_internal_native_chat_format",
            return_value="auto",
        ):
            prefer_gguf_jinja_chat_format(llama)
        self.assertEqual(llama.chat_format, "chatml")

    def test_respects_explicit_user_chat_format_choice(self) -> None:
        llama = _LlamaStub()
        with patch(
            "core.native_llama_chat.get_internal_native_chat_format",
            return_value="chatml",
        ):
            prefer_gguf_jinja_chat_format(llama)
        self.assertEqual(llama.chat_format, "llama-2")


if __name__ == "__main__":
    unittest.main()
