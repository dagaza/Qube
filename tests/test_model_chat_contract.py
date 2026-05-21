"""Tests for core/model_chat_contract load-time binding (no llama-cpp required)."""

from __future__ import annotations

import unittest

from core.model_chat_contract import (
    ChatContract,
    build_model_info_from_llama,
    chat_contract_to_resolution,
    map_chat_contract_to_llama_chat_format,
    resolve_chat_contract,
)


class ResolveChatContractTests(unittest.TestCase):
    def test_order_template_plus_handler_is_gguf(self) -> None:
        mi = {
            "model_display_name": "TestModel",
            "model_basename": "t.gguf",
            "tokenizer_chat_template": "{{ bos_token }}",
            "chat_handler_keys": ["chat_template.default", "chatml"],
            "metadata": {"tokenizer.chat_template": "{{ bos_token }}"},
        }
        c = resolve_chat_contract(mi)
        self.assertEqual(c.format_name, "chat_template.default")
        self.assertEqual(c.source, "gguf")
        self.assertIn("tokenizer.chat_template", " ".join(c.binding_reasoning))

    def test_oss_resolves_chatml_not_llama2(self) -> None:
        for name in ("gpt-oss-20b", "my-oss-model", "Llama-OSS-v1"):
            mi = {
                "model_display_name": name,
                "model_basename": "x.gguf",
                "tokenizer_chat_template": None,
                "chat_handler_keys": [],
                "metadata": {},
            }
            c = resolve_chat_contract(mi)
            self.assertEqual(
                c.format_name,
                "chatml",
                msg=f"expected chatml for {name!r}, got {c.format_name}",
            )
            self.assertNotEqual(c.format_name, "llama-2")

    def test_unknown_fallback_chatml(self) -> None:
        mi = {
            "model_display_name": "ZebraUnknownXYZ",
            "model_basename": "z.gguf",
            "tokenizer_chat_template": None,
            "chat_handler_keys": ["chatml", "llama-2"],
            "metadata": {},
        }
        c = resolve_chat_contract(mi)
        res = chat_contract_to_resolution(c)
        self.assertEqual(c.format_name, "chatml")
        self.assertEqual(c.source, "fallback")
        self.assertTrue(res.fallback_used)

    def test_stability_same_model_info_twice(self) -> None:
        mi = {
            "model_display_name": "StableM",
            "model_basename": "s.gguf",
            "tokenizer_chat_template": None,
            "chat_handler_keys": ["chatml"],
            "metadata": {},
        }
        a = resolve_chat_contract(mi)
        b = resolve_chat_contract(mi)
        self.assertEqual(a.format_name, b.format_name)
        self.assertEqual(a.source, b.source)
        self.assertEqual(a.model_name, b.model_name)

    def test_map_downgrades_unsupported_handler(self) -> None:
        chat = ChatContract(
            model_name="m",
            format_name="llama-3",
            tokenizer_template=None,
            source="hf_config",
            locked=True,
            binding_reasoning=[],
        )
        out = map_chat_contract_to_llama_chat_format(
            chat, handler_keys={"chatml", "llama-2"}
        )
        self.assertEqual(out, "chatml")


class BuildModelInfoFromLlamaTests(unittest.TestCase):
    def test_build_model_info_from_llama_shape(self) -> None:
        class _Fake:
            metadata = {
                "tokenizer.chat_template": "abc",
                "general.name": "My Model",
            }
            _chat_handlers = {"chatml": object(), "chat_template.default": object()}

        info = build_model_info_from_llama(llama=_Fake(), model_path="/tmp/foo.gguf")
        self.assertEqual(info["model_basename"], "foo.gguf")
        self.assertEqual(info["model_display_name"], "My Model")
        self.assertEqual(info["tokenizer_chat_template"], "abc")
        self.assertIn("chatml", info["chat_handler_keys"])
        self.assertIn("chat_template.default", info["chat_handler_keys"])


if __name__ == "__main__":
    unittest.main()
