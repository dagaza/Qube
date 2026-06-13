"""Tests for Harmony protocol detection and contract metadata."""
from __future__ import annotations

import unittest

from core.harmony_protocol import (
    detect_harmony_protocol,
    harmony_stops_for_contract,
    is_harmony_contract,
    is_harmony_model_name,
    template_indicates_harmony,
)
from core.prompt_contract import PromptContract, resolve_prompt_contract


class _FakeLlama:
    def __init__(self, *, name: str, metadata: dict | None = None, handlers: dict | None = None):
        self.metadata = {"general.name": name}
        if metadata:
            self.metadata.update(metadata)
        self._chat_handlers = handlers or {}
        self.chat_format = "llama-2"
        self.model_path = f"/tmp/{name}.gguf"


class TestHarmonyProtocol(unittest.TestCase):
    def test_name_detection(self) -> None:
        self.assertTrue(is_harmony_model_name("gpt-oss-20b"))
        self.assertTrue(is_harmony_model_name("My-GPT-OSS-v1"))
        self.assertFalse(is_harmony_model_name("llama-3-8b"))

    def test_template_detection(self) -> None:
        tmpl = (
            "{{ '<|start|>' }}{{ role }}<|message|>{{ content }}<|end|>"
            "<|channel|>final"
        )
        self.assertTrue(template_indicates_harmony(tmpl))

    def test_detect_from_metadata_architecture(self) -> None:
        prof = detect_harmony_protocol(
            model_name="some-quant",
            metadata={"general.architecture": "gptoss"},
        )
        self.assertIsNotNone(prof)
        self.assertEqual(prof.detection_method, "metadata")

    def test_resolve_contract_sets_protocol(self) -> None:
        llama = _FakeLlama(name="gpt-oss-20b", handlers={"chatml": object()})
        c = resolve_prompt_contract(llama, [{"role": "user", "content": "Hi"}]).contract
        self.assertEqual(c.protocol, "harmony")
        self.assertIn("<|return|>", c.stop)

    def test_harmony_stops_include_phrase_stops_by_default(self) -> None:
        default = harmony_stops_for_contract()
        narrow = harmony_stops_for_contract(include_phrase_stops=False)
        self.assertIn("<|return|>", default)
        self.assertGreater(len(default), 1)
        self.assertEqual(narrow, ["<|return|>"])

    def test_is_harmony_contract_by_protocol_field(self) -> None:
        c = PromptContract(
            mode="rendered",
            chat_format=None,
            prompt="x",
            messages=None,
            stop=["<|return|>"],
            template_source="fallback",
            confidence="high",
            protocol="harmony",
        )
        self.assertTrue(is_harmony_contract(c))

    def test_harmony_model_active_from_contract(self) -> None:
        from core.harmony_protocol import harmony_model_active

        c = PromptContract(
            mode="rendered",
            chat_format=None,
            prompt="x",
            messages=None,
            stop=["<|return|>"],
            template_source="fallback",
            confidence="high",
            protocol="harmony",
        )
        self.assertTrue(harmony_model_active(contract=c))
        self.assertFalse(harmony_model_active(model_name="gemma-4-26b"))
        self.assertTrue(harmony_model_active(model_name="gpt-oss-20b"))
