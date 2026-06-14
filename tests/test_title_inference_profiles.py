"""Titling inference profile and instrumentation tests (no llama required)."""
from __future__ import annotations

import unittest

from core.sidecar_prompts import build_title_task_parts, instrument_title_parse
from core.title_generation_experiment import aggregate_title_experiment_metrics
from core.title_inference_profiles import (
    PROFILE_IDS,
    get_title_profile,
    normalize_title_context_mode,
    normalize_title_inference_profile,
)


class TestTitleInferenceProfiles(unittest.TestCase):
    def test_profile_ids(self) -> None:
        self.assertEqual(PROFILE_IDS, ("A", "B", "C", "D"))

    def test_profile_a_is_raw_no_think(self) -> None:
        prof = get_title_profile("A")
        self.assertEqual(prof.path, "raw")
        self.assertTrue(prof.use_no_think_directive)
        self.assertEqual(prof.temperature, 0.1)

    def test_profile_b_uses_chat_template_flag(self) -> None:
        prof = get_title_profile("B")
        self.assertEqual(prof.path, "chat")
        self.assertTrue(prof.use_enable_thinking_false)
        kw = prof.sampling_kwargs(max_tokens=32)
        self.assertEqual(kw["chat_template_kwargs"], {"enable_thinking": False})

    def test_profile_d_qwen_sampling(self) -> None:
        prof = get_title_profile("D")
        kw = prof.sampling_kwargs(max_tokens=32)
        self.assertEqual(kw["temperature"], 0.7)
        self.assertEqual(kw["top_p"], 0.8)
        self.assertEqual(kw["top_k"], 20)
        self.assertEqual(kw["min_p"], 0.0)

    def test_invalid_profile_falls_back_to_b(self) -> None:
        self.assertEqual(normalize_title_inference_profile("Z"), "B")

    def test_context_mode_normalization(self) -> None:
        self.assertEqual(normalize_title_context_mode("user"), "user_only")
        self.assertEqual(normalize_title_context_mode("full"), "full")


class TestTitleInstrumentation(unittest.TestCase):
    def test_user_only_context_excludes_assistant(self) -> None:
        _, user = build_title_task_parts(
            "Hello world",
            "Assistant says hi",
            context_mode="user_only",
        )
        self.assertIn("Hello world", user)
        self.assertNotIn("Assistant:", user)

    def test_instrument_title_parse_collects_candidates(self) -> None:
        user = (
            'Steelman both sides: "Remote work always hurts productivity"'
        )
        assistant = (
            "Remote work can indeed hurt productivity when communication is weak."
        )
        report = instrument_title_parse(
            "Remote Work Indeed Hurt Productivity",
            user_prompt=user,
            assistant_reply=assistant,
        )
        self.assertTrue(report.candidates)
        self.assertEqual(report.final_title, "Remote Work Productivity")
        self.assertTrue(report.selection.get("winner_source"))

    def test_aggregate_metrics(self) -> None:
        from core.title_generation_experiment import TitleExperimentRun

        runs = [
            TitleExperimentRun(
                profile_id="A",
                inference_ms=100.0,
                output_char_length=20,
                used_fallback_repair=True,
                think_block_stripped=True,
            ),
            TitleExperimentRun(
                profile_id="B",
                inference_ms=200.0,
                output_char_length=10,
                model_output_rejected=False,
            ),
        ]
        metrics = aggregate_title_experiment_metrics(runs)
        self.assertEqual(metrics["total_runs"], 2)
        self.assertIn("A", metrics["profiles"])
        self.assertIn("B", metrics["profiles"])


if __name__ == "__main__":
    unittest.main()
