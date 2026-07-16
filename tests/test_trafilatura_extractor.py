"""Tests for TrafilaturaExtractor (M3)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.extractors.registry import (  # noqa: E402
    extract_document,
    registered_extractors,
    select_best_extractor,
)
from core.knowledge.extractors.trafilatura_extractor import (  # noqa: E402
    EXTRACTOR_NAME,
    TrafilaturaExtractor,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "fetch"


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


class TestTrafilaturaExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import trafilatura  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("trafilatura is not installed") from exc

    def test_registered_in_registry(self) -> None:
        names = {ext.metadata.name for ext in registered_extractors()}
        self.assertIn(EXTRACTOR_NAME, names)

    def test_supports_returns_fallback_confidence(self) -> None:
        extractor = TrafilaturaExtractor()
        self.assertEqual(extractor.supports("https://x.test", "<html></html>"), 0.3)

    def test_select_best_extractor_picks_trafilatura(self) -> None:
        html = _read_fixture("article_clean.html")
        extractor, confidence = select_best_extractor("https://example.com/birds", html)
        self.assertEqual(extractor.metadata.name, EXTRACTOR_NAME)
        self.assertEqual(confidence, 0.3)

    def test_extract_document_produces_titled_sections(self) -> None:
        html = _read_fixture("article_clean.html")
        url = "https://example.com/birds"
        document = extract_document(html, url)

        self.assertEqual(document.url, url)
        self.assertIn("Dust Bathing", document.title or "")
        self.assertGreaterEqual(len(document.sections), 2)
        headings = [section.heading for section in document.sections if section.heading]
        self.assertTrue(any("dust" in (h or "").lower() for h in headings))
        self.assertIsNotNone(document.metadata)
        assert document.metadata is not None
        self.assertEqual(document.metadata.extractor_name, EXTRACTOR_NAME)
        self.assertEqual(document.metadata.extractor_version, "1.0.0")
        self.assertGreater(document.total_text_chars, 50)


if __name__ == "__main__":
    unittest.main()
