"""Tests for @-mention root menu query matching."""

from __future__ import annotations

import unittest

from core.composer_mention_trigger import (
    filter_root_row_indices,
    resolve_auto_drill_kind,
    root_row_index_for_query,
)


class TestRootRowIndexForQuery(unittest.TestCase):
    def test_empty_query_selects_first(self) -> None:
        self.assertEqual(root_row_index_for_query(""), 0)

    def test_single_letter_c_prefers_conversations(self) -> None:
        self.assertEqual(root_row_index_for_query("c"), 1)

    def test_prefix_match_on_title(self) -> None:
        self.assertEqual(root_row_index_for_query("f"), 0)
        self.assertEqual(root_row_index_for_query("fil"), 0)
        self.assertEqual(root_row_index_for_query("Files"), 0)
        self.assertEqual(root_row_index_for_query("con"), 1)
        self.assertEqual(root_row_index_for_query("Conversations"), 1)
        self.assertEqual(root_row_index_for_query("too"), 2)
        self.assertEqual(root_row_index_for_query("Tools"), 2)
        self.assertEqual(root_row_index_for_query("ski"), 3)
        self.assertEqual(root_row_index_for_query("Skills"), 3)
        self.assertEqual(root_row_index_for_query("com"), 4)
        self.assertEqual(root_row_index_for_query("Commands"), 4)

    def test_co_prefix_still_prefers_conversations(self) -> None:
        self.assertEqual(root_row_index_for_query("co"), 1)

    def test_unknown_query_falls_back_to_first(self) -> None:
        self.assertEqual(root_row_index_for_query("zzzz"), 0)


class TestFilterRootRowIndices(unittest.TestCase):
    def test_empty_query_returns_all_categories(self) -> None:
        self.assertEqual(filter_root_row_indices(""), list(range(5)))

    def test_fil_matches_files_only(self) -> None:
        self.assertEqual(filter_root_row_indices("fil"), [0])

    def test_lib_matches_file_and_tools_subtitles(self) -> None:
        matches = filter_root_row_indices("lib")
        self.assertIn(0, matches)
        self.assertIn(2, matches)

    def test_co_matches_conversations_and_commands(self) -> None:
        matches = filter_root_row_indices("co")
        self.assertIn(1, matches)
        self.assertIn(4, matches)


class TestResolveAutoDrillKind(unittest.TestCase):
    def test_empty_query_returns_none(self) -> None:
        self.assertIsNone(resolve_auto_drill_kind(""))

    def test_fil_drills_to_file(self) -> None:
        self.assertEqual(resolve_auto_drill_kind("fil"), "file")

    def test_skill_drills_to_skill(self) -> None:
        self.assertEqual(resolve_auto_drill_kind("skill"), "skill")

    def test_co_is_ambiguous(self) -> None:
        self.assertIsNone(resolve_auto_drill_kind("co"))

    def test_internet_drills_to_tool(self) -> None:
        self.assertEqual(resolve_auto_drill_kind("internet"), "tool")

    def test_lib_is_ambiguous(self) -> None:
        self.assertIsNone(resolve_auto_drill_kind("lib"))


if __name__ == "__main__":
    unittest.main()
