"""Tests for canonical trace diff UI compute helpers."""
from __future__ import annotations

import unittest

from ui.canonical_trace_diff.diff_compute import (
    diff_json_trees,
    flatten_json,
    sentence_diff_html,
    word_diff_html,
)


class DiffComputeTests(unittest.TestCase):
    def test_flatten_json_paths(self) -> None:
        flat = flatten_json({"a": {"b": 1}, "c": [2]})
        self.assertIn("a.b", flat)
        self.assertIn("c[0]", flat)

    def test_json_tree_diff_statuses(self) -> None:
        rows = diff_json_trees(
            {"temperature": 0.7, "model": "a"},
            {"temperature": 0.2, "model": "a", "stream": True},
        )
        by_path = {r["path"]: r["status"] for r in rows}
        self.assertEqual(by_path["temperature"], "modified")
        self.assertEqual(by_path["model"], "match")
        self.assertEqual(by_path["stream"], "extra")

    def test_word_diff_html_marks_changes(self) -> None:
        left, right, _ = word_diff_html("hello world", "hello there")
        self.assertIn("diff-match", left)
        self.assertIn("diff-mod", right)

    def test_sentence_diff_finds_divergence(self) -> None:
        left, right, idx = sentence_diff_html("First. Second.", "First. Other.")
        self.assertIsNotNone(idx)
        self.assertIn("divergence-marker", left)


if __name__ == "__main__":
    unittest.main()
