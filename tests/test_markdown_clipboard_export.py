import unittest

from core.citation_normalize import (
    QUBE_CITATION_HREF_PREFIX,
    markdown_for_external_clipboard,
)


class TestMarkdownClipboardExport(unittest.TestCase):
    def test_strips_internal_citation_links(self) -> None:
        md = (
            "Answer with cite "
            f"[[1]](<{QUBE_CITATION_HREF_PREFIX}1>) and "
            f"[[W]](<{QUBE_CITATION_HREF_PREFIX}W>)."
        )
        out = markdown_for_external_clipboard(md)
        self.assertEqual(out, "Answer with cite [1] and [W].")

    def test_preserves_markdown_syntax(self) -> None:
        md = "# Title\n\n**bold** and `code`"
        self.assertEqual(markdown_for_external_clipboard(md), md)


if __name__ == "__main__":
    unittest.main()
