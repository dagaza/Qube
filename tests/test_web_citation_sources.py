"""Web citation id normalization and DuckDuckGo URL extraction."""
from __future__ import annotations

import unittest

from mcp.internet_tool import _decode_ddg_target_href, _strip_html_tags
from core.citation_normalize import (
    normalize_combined_numeric_citations,
    normalize_labeled_citation_tokens,
    normalize_source_echo_citation_tokens,
)


class TestWebCitationSources(unittest.TestCase):
    def test_decode_ddg_redirect(self) -> None:
        href = (
            "https://duckduckgo.com/l/?uddg="
            "https%3A%2F%2Fexample.com%2Farticle"
        )
        self.assertEqual(_decode_ddg_target_href(href), "https://example.com/article")

    def test_strip_html_tags_unescapes(self) -> None:
        self.assertEqual(_strip_html_tags("Soak &amp; rinse"), "Soak & rinse")

    def test_labeled_web_citation_normalized_to_w(self) -> None:
        raw = "Brown rice soaks 30 minutes [W: Live Web Search]."
        self.assertEqual(
            normalize_labeled_citation_tokens(raw),
            "Brown rice soaks 30 minutes [W].",
        )

    def test_labeled_numeric_citation_normalized(self) -> None:
        raw = "See detail [2: Project Omega doc]."
        self.assertEqual(
            normalize_labeled_citation_tokens(raw),
            "See detail [2].",
        )

    def test_source_echo_multi_normalized(self) -> None:
        raw = (
            "The Knicks won 115-104 in overtime. [SOURCE 1, SOURCE 2]"
        )
        self.assertEqual(
            normalize_source_echo_citation_tokens(raw),
            "The Knicks won 115-104 in overtime. [1], [2]",
        )

    def test_source_echo_single_normalized(self) -> None:
        raw = "Final score was 115-104 [SOURCE 1]."
        self.assertEqual(
            normalize_source_echo_citation_tokens(raw),
            "Final score was 115-104 [1].",
        )

    def test_source_echo_via_labeled_pipeline(self) -> None:
        raw = "Score 115-104 [SOURCE 1, SOURCE 2]."
        self.assertEqual(
            normalize_labeled_citation_tokens(raw),
            "Score 115-104 [1], [2].",
        )

    def test_combined_numeric_citations_split(self) -> None:
        raw = "Final score 115-104 in overtime. [1, 2, 3]"
        self.assertEqual(
            normalize_combined_numeric_citations(raw),
            "Final score 115-104 in overtime. [1], [2], [3]",
        )

    def test_combined_numeric_via_full_pipeline(self) -> None:
        raw = "Knicks won. [1, 2, 3]"
        self.assertEqual(
            normalize_labeled_citation_tokens(raw),
            "Knicks won. [1], [2], [3]",
        )

    def test_single_numeric_unchanged(self) -> None:
        raw = "See [1] for details."
        self.assertEqual(normalize_combined_numeric_citations(raw), raw)


if __name__ == "__main__":
    unittest.main()
