"""Plain-text heading heuristics for Library document building."""

from __future__ import annotations

import re

_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[\.)]?\s+\S")
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s\-–—:()]{2,}$")


def _is_heading_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 100:
        return False
    if _NUMBERED_HEADING_RE.match(stripped):
        return True
    if _ALL_CAPS_HEADING_RE.match(stripped) and "  " not in stripped:
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 10:
        return True
    words = stripped.split()
    if 1 <= len(words) <= 8 and stripped.istitle():
        return True
    return False


def split_plain_text_sections(text: str) -> list[tuple[str | None, int, str]]:
    """
    Split plain text into (heading, level, body) sections.

    Uses lightweight heuristics for ALL CAPS, numbered, and title-case headings.
    """
    body = (text or "").strip()
    if not body:
        return []

    blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
    if not blocks:
        return [(None, 0, body)]

    sections: list[tuple[str | None, int, str]] = []
    current_heading: str | None = None
    current_level = 2
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_lines, current_heading, current_level
        if not current_lines:
            return
        paragraph = "\n\n".join(current_lines).strip()
        if paragraph:
            sections.append((current_heading, current_level, paragraph))
        current_lines = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) == 1 and _is_heading_line(lines[0]):
            _flush()
            current_heading = lines[0].rstrip(":")
            if _NUMBERED_HEADING_RE.match(current_heading):
                depth = current_heading.split()[0].count(".") + 1
                current_level = min(6, max(1, depth))
            else:
                current_level = 2
            continue
        current_lines.append(block)

    _flush()
    if not sections:
        return [(None, 0, body)]
    return sections
