"""Tests for composer @-mention arming helpers."""

from __future__ import annotations

import unittest

from core.composer_mention_trigger import (
    escape_strip_index,
    is_valid_mention_anchor,
    menu_trigger_strip_index,
    mention_query_suffix,
    resolve_mention_release,
)


class TestMentionAnchor(unittest.TestCase):
    def test_empty_or_whitespace_is_valid(self) -> None:
        self.assertTrue(is_valid_mention_anchor(""))
        self.assertTrue(is_valid_mention_anchor("hello "))

    def test_at_run_extension_is_valid(self) -> None:
        self.assertTrue(is_valid_mention_anchor("@"))
        self.assertTrue(is_valid_mention_anchor("@@@@"))

    def test_mid_word_is_invalid(self) -> None:
        self.assertFalse(is_valid_mention_anchor("email"))


class TestResolveMentionRelease(unittest.TestCase):
    def test_one_keystroke_opens_menu(self) -> None:
        self.assertEqual(resolve_mention_release(1), "menu")

    def test_two_or_more_keystrokes_escape(self) -> None:
        self.assertEqual(resolve_mention_release(2), "escape")
        self.assertEqual(resolve_mention_release(3), "escape")
        self.assertEqual(resolve_mention_release(4), "escape")


class TestStripIndices(unittest.TestCase):
    def test_escape_strips_trailing_at(self) -> None:
        self.assertEqual(escape_strip_index("@@", 0), 1)
        self.assertEqual(escape_strip_index("@@@", 0), 2)

    def test_menu_strips_armed_at(self) -> None:
        self.assertEqual(menu_trigger_strip_index("@", 0), 0)
        self.assertEqual(menu_trigger_strip_index("@@@@@", 4), 4)


class TestMentionQuerySuffix(unittest.TestCase):
    def test_suffix_after_armed_at(self) -> None:
        self.assertEqual(mention_query_suffix("@con", 0), "con")
        self.assertEqual(mention_query_suffix("@@@@@fil", 4), "fil")


if __name__ == "__main__":
    unittest.main()
