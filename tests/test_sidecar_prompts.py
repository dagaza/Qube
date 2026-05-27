"""Sidecar prompt parsers and contracts."""
from __future__ import annotations

import unittest

from core.sidecar_prompts import parse_task_output
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


if __name__ == "__main__":
    unittest.main()
