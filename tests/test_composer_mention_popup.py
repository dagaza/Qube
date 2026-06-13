"""Tests for @-mention root menu query matching."""

from __future__ import annotations

import unittest

from ui.components.composer_mention_popup import root_row_index_for_query


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


if __name__ == "__main__":
    unittest.main()
