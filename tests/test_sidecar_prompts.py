"""Sidecar prompt parsers and contracts."""
from __future__ import annotations

import unittest

from core.sidecar_prompts import (
    build_prompt_for_task,
    format_title_exchange_context,
    parse_task_output,
)
from core.sidecar_types import SidecarTask


class TestSidecarPrompts(unittest.TestCase):
    def test_contradiction_judge_parses_duplicate(self) -> None:
        r = parse_task_output(SidecarTask.contradiction_judge, "duplicate")
        self.assertTrue(r.ok)
        self.assertEqual(r.parsed.get("verdict"), "duplicate")

    def test_reflection_label_json(self) -> None:
        r = parse_task_output(
            SidecarTask.reflection_label,
            '{"label": "durable_user_fact"}',
        )
        self.assertEqual(r.parsed.get("label"), "durable_user_fact")

    def test_episode_summary_lines(self) -> None:
        raw = "SUMMARY: User discussed project alpha.\nTOPICS: alpha, planning"
        r = parse_task_output(SidecarTask.episode_summary, raw)
        self.assertIn("project alpha", r.parsed.get("summary", ""))
        self.assertIn("alpha", r.parsed.get("topics", []))

    def test_title_strips_quotes(self) -> None:
        r = parse_task_output(SidecarTask.title, '"Sky Color"')
        self.assertEqual(r.parsed.get("title"), "Sky Color")

    def test_title_prompt_includes_no_think_for_qwen3(self) -> None:
        prompt = build_prompt_for_task(
            SidecarTask.title,
            model_path="/models/cognition/Qwen3-1.7B-Q6_K.gguf",
            user_prompt="weather in Copenhagen",
            assistant_reply="Expect cool rain this afternoon.",
        )
        self.assertIn("/no_think", prompt)
        self.assertIn("Assistant:", prompt)
        self.assertIn("cool rain", prompt)

    def test_title_exchange_context_includes_both_turns(self) -> None:
        ctx = format_title_exchange_context(
            "What is the weather in Copenhagen today?",
            "Rain is likely with highs near 12C.",
        )
        self.assertIn("User:", ctx)
        self.assertIn("Assistant:", ctx)
        self.assertIn("Copenhagen", ctx)
        self.assertIn("Rain is likely", ctx)

    def test_title_rejects_verbatim_user_echo(self) -> None:
        user = (
            "Can you walk me through setting up a reverse proxy with nginx "
            "on Ubuntu for a small home lab?"
        )
        r = parse_task_output(
            SidecarTask.title,
            "Can you walk me through setting up a reverse proxy with nginx on Ubuntu",
            user_prompt=user,
            assistant_reply="Start by installing nginx and enabling the site config.",
        )
        self.assertTrue(r.ok)
        title = r.parsed.get("title") or ""
        self.assertNotIn("walk me through", title.lower())
        self.assertLessEqual(len(title.split()), 8)

    def test_title_fallback_prefers_assistant_topic(self) -> None:
        r = parse_task_output(
            SidecarTask.title,
            "<Think>\nplanning title\n",
            user_prompt="Hey there, can you help me with something quick?",
            assistant_reply="Quantum tunneling lets particles cross barriers classically forbidden.",
        )
        self.assertTrue(r.ok)
        title = (r.parsed.get("title") or "").lower()
        self.assertTrue(
            "quantum" in title or "tunneling" in title,
            msg=f"expected assistant topic, got {title!r}",
        )

    def test_title_strips_qwen3_think_block(self) -> None:
        raw = (
            "<Think>\nUser wants a title about the sky.\n</Think>\n"
            "blue sky"
        )
        r = parse_task_output(SidecarTask.title, raw)
        self.assertTrue(r.ok)
        self.assertEqual(r.parsed.get("title"), "Blue Sky")
        self.assertNotIn("Think", r.text)
        self.assertNotIn("\n", r.text)

    def test_title_think_only_falls_back_to_user_prompt(self) -> None:
        raw = "<Think>\nUser asked about the weather topic.\nStill planning.\n"
        r = parse_task_output(
            SidecarTask.title,
            raw,
            user_prompt="What is the weather in Copenhagen today?",
            assistant_reply="Rain and wind are expected this afternoon in Copenhagen.",
        )
        self.assertTrue(r.ok)
        title = r.parsed.get("title") or ""
        self.assertNotIn("Think", title)
        self.assertTrue(
            "weather" in title.lower()
            or "rain" in title.lower()
            or "copenhagen" in title.lower(),
        )

    def test_title_empty_model_output_uses_user_prompt(self) -> None:
        r = parse_task_output(
            SidecarTask.title,
            "",
            user_prompt="Explain quantum tunneling simply",
        )
        self.assertFalse(r.ok)

    def test_companion_line_parses_json(self) -> None:
        raw = '{"line":"Still here.","kind":"idle_quip"}'
        r = parse_task_output(SidecarTask.companion_line, raw)
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "Still here.")
        self.assertEqual(r.parsed.get("kind"), "idle_quip")

    def test_companion_line_skip(self) -> None:
        r = parse_task_output(SidecarTask.companion_line, '{"line":"","kind":"skip"}')
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "skip")

    def test_companion_line_truncates_long_line(self) -> None:
        long_line = "x" * 100
        raw = f'{{"line":"{long_line}","kind":"idle_quip"}}'
        r = parse_task_output(SidecarTask.companion_line, raw)
        self.assertTrue(r.ok)
        self.assertLessEqual(len(r.text), 72)

    def test_companion_line_v2_rewrite_prompt(self) -> None:
        from core.sidecar_prompts import build_prompt_for_task

        prompt = build_prompt_for_task(
            SidecarTask.companion_line,
            model_path="/models/cognition/Qwen3-1.7B-Q6_K.gguf",
            expression_level=2,
            thought={"intent": "wellbeing", "mood": "warm", "energy": "low"},
            observation={"type": "settings_preview", "facts": {}},
            seed_line="Still here if you need me.",
            trigger="test",
        )
        self.assertIn("seed_line", prompt)
        self.assertIn("Rephrase", prompt)
        self.assertIn("/no_think", prompt)

    def test_companion_line_rejects_low_quality_json(self) -> None:
        raw = '{"line":"Maybe something about the companion","kind":"idle_quip"}'
        r = parse_task_output(SidecarTask.companion_line, raw, trigger="idle")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "low_quality")

    def test_companion_line_partial_json_line_field(self) -> None:
        raw = '{"line":"Still here if you need me'
        r = parse_task_output(SidecarTask.companion_line, raw, trigger="test")
        self.assertTrue(r.ok)
        self.assertIn("Still here", r.text)

    def test_companion_line_rejects_tutorial_only_output(self) -> None:
        raw = '"Welcome to the Qube desktop companion, where you can customize your Qube set'
        r = parse_task_output(SidecarTask.companion_line, raw, trigger="test")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "parse_fail")

    def test_companion_line_fallback_salvages_prose(self) -> None:
        raw = (
            '"Welcome to the settings preview. Here\'s a sample: "Hello, world!"'
        )
        r = parse_task_output(
            SidecarTask.companion_line,
            raw,
            trigger="test",
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "Hello, world!")


if __name__ == "__main__":
    unittest.main()
