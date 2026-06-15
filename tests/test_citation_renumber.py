"""Tests for cited-only citation renumbering by appearance order."""
from __future__ import annotations

import unittest

from core.citation_integrity import analyze_citations, extract_citation_tokens
from core.citation_renumber import (
    extract_citation_ids_in_order,
    renumber_citations_by_appearance,
    remap_citation_ids_in_text,
)


class CitationRenumberTests(unittest.TestCase):
    def test_extract_order_first_appearance(self) -> None:
        self.assertEqual(
            extract_citation_ids_in_order("See [3] then [1] and [3] again."),
            ["3", "1"],
        )

    def test_renumber_skips_uncited_retrieval_sources(self) -> None:
        sources = [
            {"id": 1, "type": "memory", "filename": "stub"},
            {"id": 2, "type": "web", "filename": "A", "url": "https://a.test"},
            {"id": 3, "type": "web", "filename": "B", "url": "https://b.test"},
        ]
        text = "Fact from A [2]. Fact from B [3]."
        new_text, new_sources = renumber_citations_by_appearance(text, sources)
        self.assertEqual(new_text, "Fact from A [1]. Fact from B [2].")
        self.assertEqual(len(new_sources), 2)
        self.assertEqual(new_sources[0]["filename"], "A")
        self.assertEqual(new_sources[1]["filename"], "B")
        self.assertEqual(new_sources[0]["id"], 1)
        self.assertEqual(new_sources[1]["id"], 2)
        report = analyze_citations(new_text, new_sources)
        self.assertFalse(report.has_violation)

    def test_renumber_by_answer_appearance_not_retrieval_order(self) -> None:
        sources = [
            {"id": 1, "type": "web", "filename": "First"},
            {"id": 2, "type": "web", "filename": "Second"},
        ]
        text = "Uses second first [2]. Then first [1]."
        new_text, new_sources = renumber_citations_by_appearance(text, sources)
        self.assertEqual(new_text, "Uses second first [1]. Then first [2].")
        self.assertEqual([s["filename"] for s in new_sources], ["Second", "First"])

    def test_no_citations_drops_all_sources(self) -> None:
        sources = [{"id": 1, "type": "web"}, {"id": 2, "type": "web"}]
        text = "Answer with no bracket cites."
        new_text, new_sources = renumber_citations_by_appearance(text, sources)
        self.assertEqual(new_text, text)
        self.assertEqual(new_sources, [])

    def test_nested_combined_citations_renumbered(self) -> None:
        sources = [
            {"id": 1, "type": "memory", "filename": "mem"},
            {"id": 2, "type": "web", "filename": "A"},
            {"id": 3, "type": "web", "filename": "B"},
        ]
        text = "Combined [2, [3]]."
        new_text, new_sources = renumber_citations_by_appearance(text, sources)
        self.assertEqual(new_text, "Combined [1], [2].")
        self.assertEqual(len(new_sources), 2)
        tokens = extract_citation_tokens(new_text)
        self.assertEqual(tokens, {"1", "2"})

    def test_w_token_renumbered(self) -> None:
        sources = [{"id": "W", "type": "web", "filename": "Live"}]
        new_text, new_sources = renumber_citations_by_appearance("Sunny [W].", sources)
        self.assertEqual(new_text, "Sunny [1].")
        self.assertEqual(new_sources[0]["id"], 1)

    def test_remap_preserves_uncited_orphans_for_integrity_pass(self) -> None:
        sources = [{"id": 1, "type": "web"}, {"id": 2, "type": "web"}]
        text = "See [9] for detail."
        new_text, new_sources = renumber_citations_by_appearance(text, sources)
        self.assertEqual(new_text, text)
        self.assertEqual(new_sources, [])

    def test_remap_citation_ids_in_text(self) -> None:
        self.assertEqual(
            remap_citation_ids_in_text("A [2] B [3]", {"2": "1", "3": "2"}),
            "A [1] B [2]",
        )


if __name__ == "__main__":
    unittest.main()
