from __future__ import annotations

import unittest

from core.prompt_contract import (
    contains_template_markers,
    render_harmony_final_prompt,
    resolve_prompt_contract,
)


class _FakeLlama:
    def __init__(self, *, name: str, metadata: dict | None = None, handlers: dict | None = None):
        self.metadata = {"general.name": name}
        if metadata:
            self.metadata.update(metadata)
        self._chat_handlers = handlers or {}
        self.chat_format = "llama-2"
        self.model_path = f"/tmp/{name}.gguf"


class TestPromptContract(unittest.TestCase):
    def test_oss_model_routes_to_harmony_rendered_not_chatml_or_llama2(self) -> None:
        llama = _FakeLlama(name="gpt-oss-20b", handlers={"chatml": object()})
        resolved = resolve_prompt_contract(llama, [{"role": "user", "content": "Hello"}]).contract
        self.assertEqual(resolved.mode, "rendered")
        self.assertIsNone(resolved.chat_format)
        self.assertIn("<|channel|>final<|message|>", resolved.prompt or "")
        self.assertIn("<|return|>", resolved.stop)
        self.assertNotEqual(resolved.chat_format, "llama-2")
        self.assertEqual(resolved.template_source, "fallback")

    def test_oss_unsafe_gguf_template_still_uses_harmony_rendered(self) -> None:
        llama = _FakeLlama(
            name="gpt-oss-20b",
            metadata={"tokenizer.chat_template": "Hello <|channel|> world"},
            handlers={"chat_template.default": object(), "chatml": object()},
        )
        res = resolve_prompt_contract(llama, [{"role": "user", "content": "Hello"}])
        self.assertEqual(res.contract.mode, "rendered")
        self.assertIn("<|channel|>final<|message|>", res.contract.prompt or "")
        self.assertEqual(res.template_safety, {"unsafe": False, "reasons": []})

    def test_render_harmony_final_prompt_maps_messages(self) -> None:
        prompt = render_harmony_final_prompt(
            [
                {"role": "system", "content": "Be direct."},
                {"role": "user", "content": "Why is the sky blue?"},
            ]
        )
        self.assertIn("<|start|>system<|message|>Be direct.<|end|>", prompt)
        self.assertIn("<|start|>user<|message|>Why is the sky blue?<|end|>", prompt)
        self.assertTrue(prompt.endswith("<|start|>assistant<|channel|>final<|message|>"))

    def test_qwen_family_routes_to_chatml(self) -> None:
        llama = _FakeLlama(name="Qwen2.5-7B-Instruct-Q4_0", handlers={"chatml": object()})
        resolved = resolve_prompt_contract(llama, [{"role": "user", "content": "Hello"}]).contract
        self.assertEqual(resolved.chat_format, "chatml")
        self.assertEqual(resolved.template_source, "override")

    def test_unknown_model_routes_to_chatml_low_confidence(self) -> None:
        llama = _FakeLlama(name="my-custom-model", handlers={"chatml": object()})
        resolved = resolve_prompt_contract(llama, [{"role": "user", "content": "Hello"}]).contract
        self.assertEqual(resolved.chat_format, "chatml")
        self.assertEqual(resolved.confidence, "low")

    def test_gguf_template_uses_chat_template_default_when_available(self) -> None:
        llama = _FakeLlama(
            name="some-model",
            metadata={"tokenizer.chat_template": "{{ messages }}"},
            handlers={"chat_template.default": object()},
        )
        res = resolve_prompt_contract(llama, [{"role": "user", "content": "Hello"}])
        resolved = res.contract
        self.assertEqual(resolved.chat_format, "chat_template.default")
        self.assertEqual(resolved.template_source, "gguf")
        self.assertEqual(resolved.confidence, "high")
        self.assertEqual(res.template_safety, {"unsafe": False, "reasons": []})

    def test_unsafe_gguf_template_falls_back_to_chatml(self) -> None:
        llama = _FakeLlama(
            name="some-model",
            metadata={"tokenizer.chat_template": "Hello <|channel|> world"},
            handlers={"chat_template.default": object(), "chatml": object()},
        )
        res = resolve_prompt_contract(llama, [{"role": "user", "content": "Hello"}])
        c = res.contract
        self.assertEqual(c.chat_format, "chatml")
        self.assertEqual(c.template_source, "fallback_unsafe_gguf")
        self.assertEqual(c.confidence, "medium")
        assert res.template_safety is not None
        self.assertTrue(res.template_safety.get("unsafe"))
        self.assertTrue(any("channel" in str(x) for x in (res.template_safety.get("reasons") or [])))

    def test_detects_template_markers(self) -> None:
        messages = [{"role": "user", "content": "Hello <|im_start|>assistant"}]
        self.assertTrue(contains_template_markers(messages))
