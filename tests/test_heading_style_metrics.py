"""Heading-style telemetry: markdown headings vs bold-only section labels."""
from __future__ import annotations

import unittest

from core.heading_style_metrics import analyze_heading_style


class TestHeadingStyleMetrics(unittest.TestCase):
    def test_markdown_headings_only(self) -> None:
        text = "## Overview\n\nBody.\n\n### Details\n\nMore."
        metrics = analyze_heading_style(text)
        self.assertEqual(metrics.markdown_heading_count, 2)
        self.assertEqual(metrics.bold_section_title_count, 0)
        self.assertEqual(metrics.heading_style_ratio, 1.0)

    def test_bold_section_titles_only(self) -> None:
        text = (
            "**Overview**\n\n"
            "Body.\n\n"
            "**Details**\n\n"
            "More."
        )
        metrics = analyze_heading_style(text)
        self.assertEqual(metrics.markdown_heading_count, 0)
        self.assertEqual(metrics.bold_section_title_count, 2)
        self.assertEqual(metrics.heading_style_ratio, 0.0)

    def test_mixed_heading_styles_ratio(self) -> None:
        text = "## Real Heading\n\nBody.\n\n**Bold Label**\n\nMore."
        metrics = analyze_heading_style(text)
        self.assertEqual(metrics.markdown_heading_count, 1)
        self.assertEqual(metrics.bold_section_title_count, 1)
        self.assertEqual(metrics.heading_style_ratio, 0.5)

    def test_bullet_bold_labels_not_section_titles(self) -> None:
        text = "- **Temple** — historic site.\n- **Palace** — royal residence."
        metrics = analyze_heading_style(text)
        self.assertEqual(metrics.markdown_heading_count, 0)
        self.assertEqual(metrics.bold_section_title_count, 0)
        self.assertIsNone(metrics.heading_style_ratio)

    def test_plain_answer_has_no_ratio(self) -> None:
        metrics = analyze_heading_style("Paris is the capital of France.")
        self.assertEqual(metrics.markdown_heading_count, 0)
        self.assertEqual(metrics.bold_section_title_count, 0)
        self.assertIsNone(metrics.heading_style_ratio)


if __name__ == "__main__":
    unittest.main()
