"""Tests for PDF extracted-text normalization at Library ingest."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.knowledge.document.pdf_text_normalize import normalize_pdf_extracted_text


class PdfTextNormalizeTests(unittest.TestCase):
    def test_rejoins_line_break_hyphenation(self) -> None:
        self.assertEqual(normalize_pdf_extracted_text("prac-\ntice"), "practice")
        self.assertEqual(normalize_pdf_extracted_text("diffi-\ncult"), "difficult")
        self.assertIn("in practice", normalize_pdf_extracted_text("in prac-\ntice. Some succeeded"))

    def test_keeps_intentional_hyphens(self) -> None:
        self.assertEqual(
            normalize_pdf_extracted_text("well-known approach"),
            "well-known approach",
        )

    def test_fixes_th_ligature_splits(self) -> None:
        self.assertEqual(
            normalize_pdf_extracted_text("Th ey called it a revolution."),
            "They called it a revolution.",
        )
        self.assertEqual(
            normalize_pdf_extracted_text("Th e lesson had spread."),
            "The lesson had spread.",
        )
        self.assertEqual(normalize_pdf_extracted_text("Th ose who had fully"), "Those who had fully")

    def test_fixes_fi_and_diff_splits(self) -> None:
        self.assertEqual(normalize_pdf_extracted_text("fi rst to set"), "first to set")
        self.assertEqual(normalize_pdf_extracted_text("diffi cult life"), "difficult life")
        self.assertEqual(normalize_pdf_extracted_text("diff erently"), "differently")

    def test_collapses_letter_spaced_words(self) -> None:
        self.assertEqual(
            normalize_pdf_extracted_text("D i s c u s s i o n Questions"),
            "Discussion Questions",
        )

    def test_strips_leading_page_number_on_page_body(self) -> None:
        self.assertEqual(
            normalize_pdf_extracted_text("3\nTHE GOOD BOOK\nTh ey called"),
            "THE GOOD BOOK They called",
        )

    def test_preserves_paragraph_breaks(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        self.assertEqual(normalize_pdf_extracted_text(text), text)

    def test_book_excerpt_normalization(self) -> None:
        raw = (
            "3\nTHE GOOD BOOK\nTh ey called it a revolution. Th e lesson—the insight—had spread. "
            "in prac-\ntice. Some succeeded fully. Th ose who had fully adopted the good book's "
            "philosophy were the fi rst to set about in search of the new cheese."
        )
        cleaned = normalize_pdf_extracted_text(raw)
        self.assertIn("They called it a revolution", cleaned)
        self.assertIn("The lesson", cleaned)
        self.assertIn("in practice", cleaned)
        self.assertIn("Those who had fully", cleaned)
        self.assertIn("the first to set", cleaned)
        self.assertNotIn("Th ey", cleaned)
        self.assertNotIn("prac- tice", cleaned)


if __name__ == "__main__":
    unittest.main()
