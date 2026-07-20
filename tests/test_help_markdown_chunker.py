"""Tests for heading-aware help markdown chunking."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_markdown_chunker import chunk_help_markdown, split_help_markdown_sections


class HelpMarkdownChunkerTests(unittest.TestCase):
    def test_split_on_h2(self) -> None:
        text = "# Title\n\n## One\nBody one.\n\n## Two\nBody two."
        sections = split_help_markdown_sections(text)
        self.assertEqual(len(sections), 2)
        self.assertIn("## One", sections[0])
        self.assertIn("## Two", sections[1])

    def test_chunk_preserves_heading_boundaries(self) -> None:
        text = "## GPU layers\nAdjust offload.\n\n## Chat template\nPick a family."
        chunks = chunk_help_markdown(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("GPU layers" in chunk for chunk in chunks))
        self.assertTrue(any("Chat template" in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
