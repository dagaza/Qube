"""Tests for unified composer @-mention search."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.composer_mention_search import (
    ComposerPaletteView,
    group_search_hits,
    MentionSearchHit,
    resolve_palette_view,
    resolve_scoped_filter,
    search_composer_mentions,
    section_label,
)


class TestResolveScopedFilter(unittest.TestCase):
    def test_tools_consumed(self) -> None:
        self.assertEqual(resolve_scoped_filter("tool", "tools"), "")
        self.assertEqual(resolve_scoped_filter("tool", "tool"), "")

    def test_partial_category_not_consumed_in_scoped(self) -> None:
        self.assertEqual(resolve_scoped_filter("file", "fil"), "fil")

    def test_internet_is_item_filter(self) -> None:
        self.assertEqual(resolve_scoped_filter("tool", "internet"), "internet")


class TestResolvePaletteView(unittest.TestCase):
    def test_empty_browse(self) -> None:
        self.assertEqual(resolve_palette_view("", scoped_kind=None), ComposerPaletteView.BROWSE)

    def test_query_search(self) -> None:
        self.assertEqual(
            resolve_palette_view("inter", scoped_kind=None), ComposerPaletteView.SEARCH
        )

    def test_scoped_wins(self) -> None:
        self.assertEqual(
            resolve_palette_view("inter", scoped_kind="tool"), ComposerPaletteView.SCOPED
        )


class TestSearchComposerMentions(unittest.TestCase):
    def test_internet_finds_tool(self) -> None:
        hits = search_composer_mentions("internet", db=None, store=None)
        tool_hits = [h for h in hits if h.section == "tools"]
        self.assertTrue(any(h.label == "Internet" for h in tool_hits))

    def test_inter_alias_finds_internet(self) -> None:
        hits = search_composer_mentions("inter", db=None, store=None)
        self.assertTrue(any(h.label == "Internet" for h in hits))

    def test_web_alias_finds_internet(self) -> None:
        hits = search_composer_mentions("web", db=None, store=None)
        self.assertTrue(any(h.label == "Internet" for h in hits))

    def test_includes_category_row_for_to(self) -> None:
        hits = search_composer_mentions("to", db=None, store=None)
        self.assertTrue(any(h.section == "categories" and h.label == "Tools" for h in hits))

    def test_files_from_db(self) -> None:
        db = MagicMock()
        db.get_library_documents_for_sidebar_search.return_value = [
            {"filename": "Internet Policy.pdf", "chunk_count": 3},
        ]
        db.get_sessions_for_sidebar_search.return_value = []
        hits = search_composer_mentions("internet", db=db, store=None)
        self.assertTrue(any(h.section == "files" for h in hits))

    def test_scientific_finds_single_evidence_tool(self) -> None:
        hits = search_composer_mentions("scientific", db=None, store=None)
        tool_hits = [h for h in hits if h.section == "tools"]
        evidence = [h for h in tool_hits if h.label == "Scientific literature"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].payload.id, "evidence")

    def test_science_id_finds_alias_tool(self) -> None:
        hits = search_composer_mentions("science", db=None, store=None)
        tool_hits = [h for h in hits if h.section == "tools" and h.payload.id == "science"]
        self.assertEqual(len(tool_hits), 1)


class TestGroupSearchHits(unittest.TestCase):
    def test_section_order(self) -> None:
        hits = [
            MentionSearchHit("commands", 50, "C", "", "c"),
            MentionSearchHit("tools", 90, "T", "", "t"),
            MentionSearchHit("categories", 40, "Cat", "", "cat"),
        ]
        grouped = group_search_hits(hits)
        sections = [h.section for h in grouped]
        self.assertEqual(sections.index("categories"), 0)
        self.assertLess(sections.index("tools"), sections.index("commands"))

    def test_section_label(self) -> None:
        self.assertEqual(section_label("tools"), "Tools")


if __name__ == "__main__":
    unittest.main()
