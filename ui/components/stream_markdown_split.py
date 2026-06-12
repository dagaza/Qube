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
_MARKDOWN_FENCE_INFO = frozenset({"markdown", "md", "text", "txt"})
_PROSE_LINE = re.compile(r"^(\#{1,6}\s|(\*\*|__)|[-*+]\s|\d+\.\s)")
# Llama-family models glue headings into prose: ``...era## Prehistoric Societies In ...``
_INLINE_HEADING = re.compile(r"(?<=[^\n#])(#{1,6})\s*(?=[A-Za-z])")
_HEADER_PREFIX = re.compile(r"^(#{1,6})([A-Za-z].+)$")
_HEADER_WITH_BODY = re.compile(r"^(#{1,6}\s+)(.+)$")
_GLUED_HEADER_BODY = re.compile(r"([a-z])([A-Z][a-z]+\s+[a-z])")
_BODY_START_WORDS = (
    "In",
    "The",
    "For",
    "Early",
    "During",
    "While",
    "This",
    "These",
    "Human",
    "From",
    "One",
    "Many",
    "Most",
    "Some",
    "Such",
    "As",
    "When",
    "Where",
    "Although",
    "However",
    "After",
    "Before",
    "With",
    "By",
    "On",
    "At",
    "It",
    "They",
    "We",
    "You",
    "A",
    "An",
)
_HEADER_TITLE_BODY = re.compile(
    r"^(#{1,6}\s+)"
    r"((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))"
    r"\s+(" + "|".join(re.escape(w) for w in _BODY_START_WORDS) + r")\b"
)
_PROSE_SENTENCE_BREAK = re.compile(
    r"(\b[a-z]{4,}\b) (" + "|".join(re.escape(w) for w in _BODY_START_WORDS) + r") (?=[a-z])"
)


def normalize_inline_markdown_structure(text: str) -> str:
    """
    Repair common LLM markdown glitches before Qt parses the buffer.

    Many local models glue headings into prose (``...era.##SectionTitleBody``) so
    CommonMark never sees line-start ``##`` and the UI shows literal hash marks.
    """
    if not text:
        return text

    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        expanded = _INLINE_HEADING.sub(r"\n\n\1", line)
        for subline in expanded.split("\n"):
            if subline == "":
                out.append("")
            else:
                out.append(_normalize_prose_or_header_line(subline))
    return "\n".join(out)


def peel_unclosed_markdown_fence(text: str) -> str:
    """
    Llama-family models often wrap the whole reply in `` ```markdown `` while streaming.

    An unclosed fence makes ``split_stream_markdown_buffer`` treat the entire buffer as
    a literal tail (raw ``#`` / ``**`` on screen). Peel the wrapper so inner prose can
    render live; leave real code fences (``python``, etc.) untouched.
    """
    if not text or text.count("```") != 1:
        return text
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("```"):
        return text
    first_nl = stripped.find("\n")
    if first_nl < 0:
        return text
    info = stripped[3:first_nl].strip().lower()
    inner = stripped[first_nl + 1 :]
    if info in _MARKDOWN_FENCE_INFO:
        return inner
    if info == "" and _inner_looks_like_markdown_prose(inner):
        return inner
    return text


def split_stream_markdown_buffer(text: str) -> tuple[str, str]:
    """
    Return ``(stable_prefix, live_tail)`` maximizing ``stable_prefix`` while keeping
    the tail out of Markdown parsing until its syntax is complete.
    """
    if not text:
        return "", ""

    text = peel_unclosed_markdown_fence(text)

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


def _inner_looks_like_markdown_prose(inner: str) -> bool:
    for line in inner.splitlines():
        s = line.strip()
        if not s:
            continue
        return bool(_PROSE_LINE.match(s))
    return False


def _normalize_prose_or_header_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        m = _HEADER_PREFIX.match(stripped)
        if not m:
            return line
        normalized = f"{m.group(1)} {m.group(2)}"
        return _split_glued_header_body_line(normalized)
    return _normalize_prose_line(line)


def _normalize_prose_line(line: str) -> str:
    return _PROSE_SENTENCE_BREAK.sub(r"\1\n\n\2 ", line)


def _split_glued_header_body_line(line: str) -> str:
    stripped = line.strip()
    m = _HEADER_WITH_BODY.match(stripped)
    if not m:
        return line
    prefix, body = m.group(1), m.group(2)
    split_body = _GLUED_HEADER_BODY.sub(r"\1\n\n\2", body, count=1)
    if split_body != body:
        title_part, rest = split_body.split("\n\n", 1)
        return f"{prefix}{title_part}\n\n{_normalize_prose_line(rest)}"
    title_body = _HEADER_TITLE_BODY.match(stripped)
    if title_body:
        title = title_body.group(2)
        rest = body[len(title) :].lstrip()
        if rest:
            return f"{prefix}{title}\n\n{_normalize_prose_line(rest)}"
    return prefix + body
