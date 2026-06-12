"""
Observer-only heading-style metrics for post-inference telemetry.

Counts Markdown headings vs standalone bold-only section labels so rollout
of structured reply guidance can be tracked via ``heading_style_ratio``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)
_BOLD_SECTION_TITLE = re.compile(r"^\s*\*\*([^*][^*]*)\*\*\s*$", re.M)


@dataclass(frozen=True)
class HeadingStyleMetrics:
    markdown_heading_count: int
    bold_section_title_count: int
    heading_style_ratio: float | None

    def trace_fields(self) -> dict[str, Any]:
        return {
            "markdown_heading_count": self.markdown_heading_count,
            "bold_section_title_count": self.bold_section_title_count,
            "heading_style_ratio": self.heading_style_ratio,
        }


def analyze_heading_style(text: str) -> HeadingStyleMetrics:
    """
    Count heading styles in assistant output.

    ``heading_style_ratio`` is ``markdown / (markdown + bold_titles)`` when the
    denominator is positive; otherwise ``None`` (no section structure to score).
    """
    t = text or ""
    headings = len(_MARKDOWN_HEADING.findall(t))
    bold_titles = len(_BOLD_SECTION_TITLE.findall(t))
    denom = headings + bold_titles
    ratio = round(headings / denom, 4) if denom > 0 else None
    return HeadingStyleMetrics(
        markdown_heading_count=headings,
        bold_section_title_count=bold_titles,
        heading_style_ratio=ratio,
    )
