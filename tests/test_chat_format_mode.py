"""Chat formatting gate (Phase 2) — deterministic mode resolution."""
from __future__ import annotations

import unittest

from core.chat_format_mode import resolve_chat_format_mode
from core.harmony_renderer import render_harmony_final_prompt
from core.prompt_contract import resolve_prompt_contract


class _FakeLlama:
    def __init__(self, *, name: str):
        self.metadata = {"general.name": name}
        self._chat_handlers = {"chatml": object()}
        self.chat_format = "llama-2"
        self.model_path = f"/tmp/{name}.gguf"


class TestResolveChatFormatMode(unittest.TestCase):
    def test_factual_none_route_is_brief(self) -> None:
        mode = resolve_chat_format_mode(
            execution_route="NONE",
            user_query="What is the capital of France?",
        )
        self.assertEqual(mode, "brief")

    def test_explain_none_route_is_structured(self) -> None:
        mode = resolve_chat_format_mode(
            execution_route="NONE",
            user_query="Explain how photosynthesis works.",
        )
        self.assertEqual(mode, "structured")

    def test_rag_route_is_structured(self) -> None:
        mode = resolve_chat_format_mode(
            execution_route="RAG",
            user_query="What is the capital of France?",
        )
        self.assertEqual(mode, "structured")

    def test_ambiguous_route_is_mixed(self) -> None:
        mode = resolve_chat_format_mode(
            execution_route="CHAT",
            user_query="Thoughts on the meeting yesterday",
        )
        self.assertEqual(mode, "mixed")


class TestHarmonyBriefMode(unittest.TestCase):
    def test_brief_mode_omits_section_formatting(self) -> None:
        prompt = render_harmony_final_prompt(
            [
                {"role": "system", "content": "You are Qube."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            chat_format_mode="brief",
        )
        self.assertIn("single concise answer", prompt)
        self.assertNotIn("2–4 short sections", prompt)

    def test_gpt_oss_chat_brief_via_contract(self) -> None:
        llama = _FakeLlama(name="gpt-oss-20b")
        contract = resolve_prompt_contract(
            llama,
            [
                {"role": "system", "content": "You are Qube."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            task="chat",
            chat_format_mode="brief",
        ).contract
        prompt = contract.prompt or ""
        self.assertNotIn("2–4 short sections", prompt)
        self.assertIn("single concise answer", prompt)


if __name__ == "__main__":
    unittest.main()
