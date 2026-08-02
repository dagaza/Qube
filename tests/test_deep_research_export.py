"""Tests for deep-research report export helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research_export import (  # noqa: E402
    extract_research_query,
    format_research_report_for_export,
    is_deep_research_report,
    suggested_research_export_stem,
    write_research_report_markdown,
)

_SAMPLE_REPORT = """# Deep Research Report

**Query:** ACE inhibitors heart failure

## Findings

Evidence summary here.
"""


class TestDeepResearchExport(unittest.TestCase):
    def test_is_deep_research_report(self) -> None:
        self.assertTrue(is_deep_research_report(_SAMPLE_REPORT))
        self.assertFalse(is_deep_research_report("Hello world"))

    def test_extract_research_query(self) -> None:
        self.assertEqual(
            extract_research_query(_SAMPLE_REPORT),
            "ACE inhibitors heart failure",
        )

    def test_format_adds_export_footer(self) -> None:
        body = format_research_report_for_export(_SAMPLE_REPORT)
        self.assertIn("_Exported ", body)
        self.assertIn("_Query: ACE inhibitors heart failure_", body)

    def test_suggested_stem_uses_query(self) -> None:
        stem = suggested_research_export_stem(query="ACE inhibitors HF")
        self.assertTrue(stem.startswith("research_ACE inhibitors HF_"))

    def test_write_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = write_research_report_markdown(
                _SAMPLE_REPORT,
                Path(tmp) / "report",
            )
            self.assertTrue(dest.exists())
            self.assertEqual(dest.suffix, ".md")
            text = dest.read_text(encoding="utf-8")
            self.assertIn("# Deep Research Report", text)


if __name__ == "__main__":
    unittest.main()
