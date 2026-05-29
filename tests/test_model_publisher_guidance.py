"""Tests for deterministic README publisher guidance extraction and merge."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.model_publisher_guidance import (
    PublisherGuidance,
    apply_guidance_to_reasoning_profile,
    extract_publisher_guidance,
    lookup_curated_publisher_guidance,
    merge_publisher_guidance,
)
from core.model_reasoning_profile import ModelReasoningProfile
from core.publisher_guidance_service import PublisherGuidanceService
from core.prompt_template_router import build_prompt_bundle
from core.execution_policy import ExecutionPolicy
from core.system_capabilities_store import SystemCapabilitiesStore


EXAMPLE_README = """
System Role / System Prompts - Reasoning On/Off/Variable and Augment The Model's Power:

System Role / System Prompt / System Message is "root access" to the model and controls
internal workings - both instruction following and output generation and in the case of
this model reasoning control and on/off for reasoning too.

If you do not set a "system prompt", reasoning/thinking will be OFF by default, and the
model will operate like a normal LLM.

HOW TO SET:

In Koboldcpp, load the model, start it, go to settings -> select "Llama 3 Chat"/"Command-R"
and enter the text in the "sys prompt" box.

SIMPLE:

You are a helpful, smart, kind, and efficient AI assistant.

MULTI-TIERED [reasoning on]:

You are a deep thinking AI composed of 4 AIs - Spock, Wordsmith, Jamet and Saten, -
you may use extremely long chains of thought to deeply consider the problem and
deliberate with yourself via systematic reasoning processes to help come to a correct
solution prior to answering. You should enclose your thoughts and internal monologue
inside <think> </think> tags, and then provide your solution.

CREATIVE MULTI-TIERED [reasoning on]:

Below is an instruction that describes a task. Ponder each user instruction carefully.

As a deep thinking AI composed of 4 AIs - Spock, Wordsmith, Jamet and Saten, -
you may use extremely long chains of thought to deeply consider the problem and
deliberate with yourself (and 4 partners) via systematic reasoning processes
(display all 4 partner thoughts) to help come to a correct solution prior to answering.
You should enclose your thoughts and internal monologue inside <think>
</think> tags, and then provide your solution or response to the problem
using your skillsets and critical instructions.

Here are your skillsets:
[MASTERSTORY]:NarrStrct(StryPlnng,Strbd,ScnSttng,Exps,Dlg,Pc)-CharDvlp(ChrctrCrt,ChrctrArcs)

Here are your critical instructions:
Ponder each word choice carefully to present as vivid and emotional journey as is possible.
"""


class TestExtractPublisherGuidance(unittest.TestCase):
    def test_full_example_readme(self) -> None:
        g = extract_publisher_guidance(EXAMPLE_README)
        self.assertIsNotNone(g)
        assert g is not None
        self.assertIn("<think>", g.thinking_tags)
        self.assertIn("</think>", g.thinking_tags)
        self.assertEqual(g.default_reasoning_without_system, "off")
        self.assertTrue(g.reasoning_controlled_by_system)
        self.assertIn("llama3", g.mentioned_chat_templates)
        self.assertIn("mistral", g.mentioned_chat_templates)
        self.assertTrue(any("ignored_preset_block" in e for e in g.evidence))
        self.assertEqual(g.source, "readme")

    def test_empty_readme_returns_none(self) -> None:
        self.assertIsNone(extract_publisher_guidance(""))
        self.assertIsNone(extract_publisher_guidance("   "))

    def test_quant_only_readme_returns_none(self) -> None:
        self.assertIsNone(
            extract_publisher_guidance("Download Q4_K_M or Q8_0 from the files tab.")
        )

    def test_creative_block_no_system_prompt_field(self) -> None:
        g = extract_publisher_guidance(EXAMPLE_README)
        self.assertIsNotNone(g)
        d = g.to_dict() if g else {}
        self.assertNotIn("system_prompt", d)


class TestCuratedAndMerge(unittest.TestCase):
    def test_curated_pattern_on_model_id(self) -> None:
        registry = {
            "publisher_guidance": {
                "patterns": [
                    {
                        "match": "redacted",
                        "type": "contains",
                        "guidance": {
                            "thinking_tags": ["<think>", "</think>"],
                            "default_reasoning_without_system": "off",
                            "reasoning_controlled_by_system": True,
                        },
                    }
                ]
            }
        }
        g = lookup_curated_publisher_guidance(
            registry,
            model_id="user/My-Redacted-Thinking-GGUF",
            normalized_model_id="my-redacted-thinking-gguf",
        )
        self.assertIsNotNone(g)
        assert g is not None
        self.assertEqual(g.source, "curated_pattern")

    def test_curated_beats_weaker_readme(self) -> None:
        readme_g = extract_publisher_guidance(EXAMPLE_README)
        curated = PublisherGuidance(
            thinking_tags=("<think>", "</think>"),
            default_reasoning_without_system="off",
            reasoning_controlled_by_system=True,
            mentioned_chat_templates=(),
            confidence=0.9,
            source="curated",
            evidence=("curated:exact",),
        )
        merged = merge_publisher_guidance(readme_g, curated)
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.source, "curated")

    def test_apply_guidance_boosts_heuristic_profile(self) -> None:
        profile = ModelReasoningProfile(
            model_name="test",
            supports_thinking_tokens=False,
            thinking_token_patterns=[],
            default_mode="direct",
            reasoning_confidence=0.3,
            detection_method="fallback",
        )
        guidance = PublisherGuidance(
            thinking_tags=("<think>",),
            default_reasoning_without_system="off",
            reasoning_controlled_by_system=True,
            mentioned_chat_templates=(),
            confidence=0.7,
            source="readme",
            evidence=("thinking_tags",),
        )
        boosted = apply_guidance_to_reasoning_profile(profile, guidance)
        self.assertTrue(boosted.supports_thinking_tokens)
        self.assertIn("<think>", boosted.thinking_token_patterns)
        self.assertIn("readme_guidance", boosted.detection_method)

    def test_apply_guidance_skips_tokenizer_scan(self) -> None:
        profile = ModelReasoningProfile(
            model_name="test",
            supports_thinking_tokens=True,
            thinking_token_patterns=["<think>"],
            default_mode="thinking",
            reasoning_confidence=1.0,
            detection_method="tokenizer_scan",
        )
        guidance = PublisherGuidance(
            thinking_tags=("<thinking>",),
            default_reasoning_without_system="off",
            reasoning_controlled_by_system=False,
            mentioned_chat_templates=(),
            confidence=0.7,
            source="readme",
            evidence=(),
        )
        same = apply_guidance_to_reasoning_profile(profile, guidance)
        self.assertEqual(same.detection_method, "tokenizer_scan")


class TestPublisherGuidanceService(unittest.TestCase):
    def test_provenance_lookup_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SystemCapabilitiesStore(system_data_dir=td)
            svc = PublisherGuidanceService(store=store)
            repo = "publisher/test-redacted-thinking-GGUF"
            svc.extract_and_store(repo, EXAMPLE_README)
            local = "/tmp/models/test-model.gguf"
            svc.record_provenance(local, repo)
            g = svc.lookup_for_load(local, "test-model", repo_id=None)
            self.assertIsNotNone(g)
            assert g is not None
            self.assertIn("<think>", g.thinking_tags)


class TestBuildPromptBundleGuidance(unittest.TestCase):
    def test_merge_thinking_tags_into_stops(self) -> None:
        llama = MagicMock()
        llama.metadata = {"tokenizer.chat_template": "<|im_start|>user"}
        llama.chat_format = "chatml"
        llama.model_path = "/tmp/test.gguf"

        guidance = PublisherGuidance(
            thinking_tags=("<think>", "</think>"),
            default_reasoning_without_system="off",
            reasoning_controlled_by_system=True,
            mentioned_chat_templates=(),
            confidence=0.8,
            source="readme",
            evidence=(),
        )
        pol = ExecutionPolicy(
            execution_mode="direct",
            allow_thinking_tokens=False,
            strip_thinking_output=True,
            ui_display_thinking=False,
            tts_strip_thinking=True,
            enforcement_mode="hard",
        )

        with patch("core.prompt_template_router.reconstruct_formatted_prompt") as recon:
            recon.return_value = ("<|im_start|>user\nhi<|im_start|>assistant\n", [], "")
            with patch("core.prompt_template_router.native_chat_completion_kwargs") as nck:
                nck.return_value = {"stop": []}
                bundle, _, _ = build_prompt_bundle(
                    llama,
                    [{"role": "user", "content": "hi"}],
                    None,
                    pol,
                    publisher_guidance=guidance,
                )
        stops = bundle.stop_tokens
        self.assertIn("<think>", stops)
        self.assertIn("</think>", stops)


if __name__ == "__main__":
    unittest.main()
