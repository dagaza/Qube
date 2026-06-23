from __future__ import annotations

import unittest
from unittest.mock import patch

from core.execution_policy import ExecutionPolicy
from core.model_reasoning_profile import ModelReasoningProfile
from core.prompt_template_router import (
    apply_reasoning_injection,
    build_prompt_bundle,
    infer_template_type,
)


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

    def test_mistral_anchor_does_not_append_phi_assistant_token(self) -> None:
        llama = _F()
        llama.chat_format = "mistral-instruct"
        pol = ExecutionPolicy(
            execution_mode="direct",
            allow_thinking_tokens=False,
            strip_thinking_output=True,
            ui_display_thinking=False,
            tts_strip_thinking=True,
            enforcement_mode="hard",
        )
        prof = ModelReasoningProfile(
            model_name="Mistral-7B-Instruct-v0.3",
            supports_thinking_tokens=False,
            thinking_token_patterns=[],
            default_mode="direct",
            reasoning_confidence=0.5,
            detection_method="test",
        )
        prompt = (
            "[INST] You are Qube.\n"
            "Why soak brown rice? [/INST]"
        )
        with patch(
            "core.prompt_template_router.reconstruct_formatted_prompt",
            return_value=(prompt, ["</s>"], "n"),
        ), patch(
            "core.prompt_template_router.infer_template_type",
            return_value="mistral",
        ), patch(
            "core.prompt_template_router.detect_template_override",
            return_value=None,
        ), patch(
            "core.prompt_template_router.get_override",
        ) as mock_ov:
            from core.model_override_store import LearnedOverride

            mock_ov.return_value = LearnedOverride(
                model_name="Mistral-7B-Instruct-v0.3",
                force_execution_mode=None,
                enforcement_mode=None,
                strip_thinking=None,
                extra_stop_tokens=["</s>"],
                enforce_assistant_anchor=True,
            )
            bundle, _note, _ = build_prompt_bundle(
                llama,
                [{"role": "user", "content": "Why soak brown rice?"}],
                prof,
                pol,
            )
        self.assertNotIn("<|assistant|>", bundle.prompt)
        self.assertIn("Write only the user-facing response.", bundle.prompt)
        self.assertTrue(bundle.prompt.rstrip().endswith("[/INST]") is False)

    def test_gemma_disabled_reasoning_injection_before_turn_model_anchor(self) -> None:
        prompt = (
            "<bos><|turn>system\nYou are Qube.<turn|>\n"
            "<|turn>user\nWhy is the sky blue?<turn|>\n"
            "<|turn>model\n"
        )
        out = apply_reasoning_injection(prompt, "gemma", "disabled")
        self.assertIn("Write only the user-facing response.", out)
        self.assertLess(out.index("Write only the user-facing response."), out.rindex("<|turn>model"))
        self.assertNotIn("<|assistant|>", out)
        self.assertTrue(out.rstrip().endswith("<|turn>model"))

    def test_infer_template_type_detects_gemma_turn_markers(self) -> None:
        llama = _F()
        llama.metadata = {
            "tokenizer.chat_template": "{{ '<|turn>user' }}",
            "general.name": "other-model",
        }
        self.assertEqual(infer_template_type(llama), "gemma")

    def test_nemotron_skips_reasoning_injection_when_disabled(self) -> None:
        llama = _F()
        llama.metadata = {"general.name": "NVIDIA-Nemotron-3-Nano-4B"}
        llama.model_path = "/models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
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
        ) as mock_r:
            bundle, _note, _ = build_prompt_bundle(
                llama,
                [{"role": "user", "content": "Why is the sky blue?"}],
                None,
                pol,
                effective_chat_format="chat_template.default",
                prompt_contract_stops=["<|im_end|>"],
            )
        self.assertEqual(bundle.prompt, prompt)
        self.assertNotIn("Write only the user-facing response.", bundle.prompt)
        _args, kwargs = mock_r.call_args
        self.assertEqual(kwargs.get("chat_template_kwargs"), {"enable_thinking": False})

    def test_nemotron_jinja_filters_thinking_stops_but_keeps_template_markers(self) -> None:
        llama = _F()
        llama.chat_format = "chat_template.default"
        llama.metadata = {
            "general.name": "NVIDIA-Nemotron-3-Nano-4B",
            "tokenizer.chat_template": (
                "<|im_start|>assistant\n<|assistant|>\n</|assistant|>"
            ),
        }
        llama.model_path = "/models/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"
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
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n<|assistant|>\n"
        )
        with patch(
            "core.prompt_template_router.reconstruct_formatted_prompt",
            return_value=(prompt, ["<|im_end|>"], "n"),
        ):
            bundle, _note, _ = build_prompt_bundle(
                llama,
                [{"role": "user", "content": "What is 2+2?"}],
                None,
                pol,
                effective_chat_format="chat_template.default",
                prompt_contract_stops=["<|im_end|>"],
            )
        self.assertIn("<|im_end|>", bundle.stop_tokens)
        self.assertIn("<|assistant|>", bundle.stop_tokens)
        self.assertIn("</|assistant|>", bundle.stop_tokens)
        self.assertNotIn("<think>", bundle.stop_tokens)
        self.assertNotIn("</think>", bundle.stop_tokens)


if __name__ == "__main__":
    unittest.main()
