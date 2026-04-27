from __future__ import annotations

import unittest
from unittest.mock import patch

from core.execution_policy import ExecutionPolicy
from core.model_reasoning_profile import ModelReasoningProfile
from core.prompt_template_router import build_prompt_bundle


class _F:
    def __init__(self) -> None:
        self.metadata: dict = {}
        self.chat_format = "chatml"
        self.model_path = "/x.gguf"
        self._chat_handlers = {"chatml": object()}


class TestBuildPromptBundle(unittest.TestCase):
    def test_forwards_reconstruct_kwargs_and_merges_contract_stops(self) -> None:
        llama = _F()
        pol = ExecutionPolicy(
            execution_mode="direct",
            allow_thinking_tokens=False,
            strip_thinking_output=True,
            ui_display_thinking=False,
            tts_strip_thinking=True,
            enforcement_mode="soft",
        )
        prof = ModelReasoningProfile(
            model_name="unit",
            supports_thinking_tokens=False,
            thinking_token_patterns=[],
            default_mode="direct",
            reasoning_confidence=0.5,
            detection_method="test",
        )
        with patch(
            "core.prompt_template_router.reconstruct_formatted_prompt",
            return_value=("PROMPT", ["fmt"], "n"),
        ) as mock_r:
            bundle, note, _ = build_prompt_bundle(
                llama,
                [{"role": "user", "content": "hi"}],
                prof,
                pol,
                effective_chat_format="chatml",
                suppress_gguf_metadata=True,
                prompt_contract_stops=["<|im_end|>"],
            )
        self.assertIn("PROMPT", bundle.prompt)
        mock_r.assert_called_once()
        _args, kwargs = mock_r.call_args
        self.assertEqual(kwargs.get("effective_chat_format"), "chatml")
        self.assertTrue(kwargs.get("suppress_gguf_metadata"))
        self.assertIn("<|im_end|>", bundle.stop_tokens)
        self.assertEqual(note, "n")

    def test_disabled_reasoning_injection_precedes_chatml_assistant_anchor(self) -> None:
        llama = _F()
        pol = ExecutionPolicy(
            execution_mode="direct",
            allow_thinking_tokens=False,
            strip_thinking_output=True,
            ui_display_thinking=False,
            tts_strip_thinking=True,
            enforcement_mode="soft",
        )
        prompt = (
            "<|im_start|>system\nYou are Qube.<|im_end|>\n"
            "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        with patch(
            "core.prompt_template_router.reconstruct_formatted_prompt",
            return_value=(prompt, ["<|im_end|>"], "n"),
        ):
            bundle, _note, _ = build_prompt_bundle(
                llama,
                [{"role": "user", "content": "Why is the sky blue?"}],
                None,
                pol,
                effective_chat_format="chatml",
                prompt_contract_stops=["<|im_end|>"],
            )
        injection_idx = bundle.prompt.index("Write only the user-facing response.")
        assistant_idx = bundle.prompt.rindex("<|im_start|>assistant")
        self.assertLess(injection_idx, assistant_idx)


if __name__ == "__main__":
    unittest.main()
