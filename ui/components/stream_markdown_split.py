"""
Split an in-progress assistant buffer into markdown-safe prefix + live tail.

While streaming, only the stable prefix is parsed as Markdown; the tail is escaped
so digits, list markers, and partial ``**`` spans render literally until complete.
"""
from __future__ import annotations

import re

_ORDERED_LIST_LINE = re.compile(r"^\s*\d+\.\s")
_TABLE_ROW = re.compile(r"^\s*\|")
_LINK_DEST = re.compile(r"\]\([^\)]*$")


def split_stream_markdown_buffer(text: str) -> tuple[str, str]:
    """
    Return ``(stable_prefix, live_tail)`` maximizing ``stable_prefix`` while keeping
    the tail out of Markdown parsing until its syntax is complete.
    """
    if not text:
        return "", ""

    split = len(text)
    split = min(split, _split_before_unclosed_delimiter(text, "**"))
    split = min(split, _split_before_unclosed_delimiter(text, "__"))
    split = min(split, _split_before_unclosed_inline_code(text))
    split = min(split, _split_before_unclosed_link(text))
    split = min(split, _split_before_incomplete_list_line(text))
    split = min(split, _split_before_incomplete_table(text))
    split = min(split, _split_before_incomplete_fence(text))

    return text[:split], text[split:]


def compose_streaming_markdown(stable: str, tail: str) -> str:
    """Single ``setMarkdown`` input: formatted prefix + literal tail."""
    if not tail:
        return stable
    if not stable:
        return escape_markdown_literal(tail)
    return stable + escape_markdown_literal(tail)


def escape_markdown_literal(text: str) -> str:
    """Backslash-escape tail text so Qt's Markdown parser treats it as plain prose."""
    if not text:
        return ""

    lines: list[str] = []
    for line in text.split("\n"):
        escaped: list[str] = []
        for ch in line:
            if ch == "\\":
                escaped.append("\\\\")
            elif ch in "*_`#[]()|<>!~":
                escaped.append("\\" + ch)
            else:
                escaped.append(ch)
        fixed = "".join(escaped)
        fixed = re.sub(r"^(\s*)(\d+)\.", r"\1\2\\.", fixed)
        lines.append(fixed)
    return "\n".join(lines)


def _split_before_unclosed_delimiter(text: str, delim: str) -> int:
    if text.count(delim) % 2 == 0:
        return len(text)
    pos = text.rfind(delim)
    return pos if pos >= 0 else len(text)


def _split_before_unclosed_inline_code(text: str) -> int:
    """Hold tail when a single-backtick span is open (ignore ``` fences)."""
    i = 0
    n = len(text)
    in_fence = False
    tick_open: int | None = None
    while i < n:
        if text.startswith("```", i):
            in_fence = not in_fence
            i += 3
            continue
        if in_fence:
            i += 1
            continue
        if text[i] == "`":
            if tick_open is None:
                tick_open = i
            else:
                tick_open = None
        i += 1
    if tick_open is None:
        return n
    return tick_open


def _split_before_unclosed_link(text: str) -> int:
    """Hold ``[label](url`` until the destination closes."""
    m = _LINK_DEST.search(text)
    if not m:
        return len(text)
    return m.start() + 1


def _split_before_incomplete_list_line(text: str) -> int:
    nl = text.rfind("\n")
    line_start = nl + 1 if nl >= 0 else 0
    line = text[line_start:]
    if not _ORDERED_LIST_LINE.match(line):
        return len(text)
    if line.count("**") % 2 == 1:
        return line_start
    if not text.endswith("\n"):
        return line_start
    return len(text)


def _split_before_incomplete_table(text: str) -> int:
    nl = text.rfind("\n")
    line_start = nl + 1 if nl >= 0 else 0
    line = text[line_start:]
    if not _TABLE_ROW.match(line):
        return len(text)
    if not text.endswith("\n"):
        return line_start
    return len(text)


def _split_before_incomplete_fence(text: str) -> int:
    if text.count("```") % 2 == 0:
        return len(text)
    pos = text.rfind("```")
    return pos if pos >= 0 else len(text)
