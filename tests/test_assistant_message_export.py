"""Tests for assistant message export helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.assistant_message_export import (  # noqa: E402
    format_assistant_message_for_export,
    has_exportable_assistant_content,
    suggested_assistant_export_stem,
    write_assistant_message_markdown,
)

_RESEARCH_REPORT = """# Deep Research Report

**Query:** ACE inhibitors heart failure

## Findings

Evidence summary here.
"""


class TestAssistantMessageExport(unittest.TestCase):
    def test_has_exportable_assistant_content(self) -> None:
        self.assertFalse(has_exportable_assistant_content(""))
        self.assertFalse(has_exportable_assistant_content("   "))
        self.assertTrue(has_exportable_assistant_content("Hello world"))

    def test_generic_format_adds_export_footer(self) -> None:
        body = format_assistant_message_for_export("## Summary\n\nPlain answer.")
        self.assertIn("Plain answer.", body)
        self.assertIn("_Exported ", body)

    def test_research_report_uses_research_formatter(self) -> None:
        body = format_assistant_message_for_export(_RESEARCH_REPORT)
        self.assertIn("# Deep Research Report", body)
        self.assertIn("_Query: ACE inhibitors heart failure_", body)

    def test_suggested_stem_for_generic_answer(self) -> None:
        stem = suggested_assistant_export_stem("## Quantum computing basics\n\nText.")
        self.assertTrue(stem.startswith("answer_Quantum computing basics_"))

    def test_suggested_stem_for_research_report(self) -> None:
        stem = suggested_assistant_export_stem(_RESEARCH_REPORT)
        self.assertTrue(stem.startswith("research_ACE inhibitors heart failure_"))

    def test_write_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = write_assistant_message_markdown(
                "A short assistant reply.",
                Path(tmp) / "answer",
            )
            self.assertTrue(dest.exists())
            self.assertEqual(dest.suffix, ".md")
            text = dest.read_text(encoding="utf-8")
            self.assertIn("A short assistant reply.", text)
            self.assertIn("_Exported ", text)


if __name__ == "__main__":
    unittest.main()
