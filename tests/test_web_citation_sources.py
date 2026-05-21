"""Web citation id normalization and DuckDuckGo URL extraction."""
from __future__ import annotations

import unittest

from mcp.internet_tool import _decode_ddg_target_href, _strip_html_tags
from core.citation_normalize import normalize_labeled_citation_tokens


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


if __name__ == "__main__":
    unittest.main()
