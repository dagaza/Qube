"""Normalize model citation tokens before Qt markdown linkification."""
from __future__ import annotations

import re

_W_LABELED_CITE = re.compile(r"\[W:\s*[^\]]+\]", re.IGNORECASE)
_NUM_LABELED_CITE = re.compile(r"\[(\d+):\s*[^\]]+\]", re.IGNORECASE)
_SOURCE_BRACKET = re.compile(r"\[([^\]]*SOURCE\s*\d+[^\]]*)\]", re.IGNORECASE)


def _replace_source_bracket(match: re.Match[str]) -> str:
    nums = re.findall(r"SOURCE\s*(\d+)", match.group(1), flags=re.IGNORECASE)
    if not nums:
        return match.group(0)
    return ", ".join(f"[{n}]" for n in nums)


def normalize_combined_numeric_citations(text: str) -> str:
    """
    Split list-style cites like ``[1, 2, 3]`` → ``[1], [2], [3]`` for Qt linkification.

    Also repairs model typos such as ``[2, [3]]`` where a nested bracket prevents
    the comma-split path from recognizing both ids.
    """
    if not text:
        return text

    def _fix_nested_typo(s: str) -> str:
        prev = None
        while s != prev:
            prev = s
            s = re.sub(
                r"\[(\d+)\s*,\s*\[(\d+)\]\]",
                r"[\1], [\2]",
                s,
            )
        return s

    text = _fix_nested_typo(text)

    def _repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if ":" in inner:
            return match.group(0)
        parts = re.split(r"\s*,\s*|\s+and\s+", inner, flags=re.IGNORECASE)
        tokens: list[str] = []
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if re.fullmatch(r"\d+", p):
                tokens.append(p)
            elif re.fullmatch(r"[wW]", p):
                tokens.append("W")
            else:
                return match.group(0)
        if len(tokens) <= 1:
            return match.group(0)
        return ", ".join("[W]" if t == "W" else f"[{t}]" for t in tokens)

    return re.sub(r"\[([^\]]+)\]", _repl, text)


def normalize_source_echo_citation_tokens(text: str) -> str:
    """
    Map model echoes of retrieval headers like ``[SOURCE 1, SOURCE 2]`` → ``[1], [2]``.
    """
    if not text:
        return text
    return _SOURCE_BRACKET.sub(_replace_source_bracket, text)


def dedupe_bookend_citation_tokens(text: str) -> str:
    """
    Drop a leading ``[N]`` when the same id appears again on that line.

    Models sometimes echo retrieval headers as both a line prefix and an
    end-of-sentence cite (``[1] Claim text. [1]``).
    """
    if not text:
        return text
    leading = re.compile(r"^(\s*)\[\s*(\d+|[wW])\s*\]\s+")
    lines: list[str] = []
    for line in text.split("\n"):
        match = leading.match(line)
        if not match:
            lines.append(line)
            continue
        ws, cid = match.group(1), match.group(2)
        rest = line[match.end() :]
        if str(cid).lower() == "w":
            dup = re.search(r"\[\s*[wW]\s*\]", rest)
        else:
            dup = re.search(rf"\[\s*{re.escape(cid)}\s*\]", rest)
        if dup:
            lines.append(f"{ws}{rest}")
        else:
            lines.append(line)
    return "\n".join(lines)


def normalize_labeled_citation_tokens(text: str) -> str:
    """
    Map echoed SOURCE headers like ``[W: Live Web Search]`` → ``[W]`` so UI
    citation linkifiers can match stored source ids.
    """
    if not text:
        return text
    t = normalize_source_echo_citation_tokens(text)
    t = _W_LABELED_CITE.sub("[W]", t)
    t = _NUM_LABELED_CITE.sub(r"[\1]", t)
    t = normalize_combined_numeric_citations(t)
    return dedupe_bookend_citation_tokens(t)


QUBE_CITATION_HREF_PREFIX = "https://qube.invalid/cite/"


def _plain_citation_token(token: str) -> str:
    return "[W]" if str(token).lower() == "w" else f"[{token}]"


def markdown_for_external_clipboard(
    md: str, *, href_prefix: str = QUBE_CITATION_HREF_PREFIX
) -> str:
    """Strip Qt-only citation link markup so pasted markdown stays Obsidian-friendly."""
    if not md:
        return ""
    esc = re.escape(href_prefix)
    out = re.sub(
        rf"\[\[(\d+|[wW])\]\]\(<{esc}\1>\)",
        lambda m: _plain_citation_token(m.group(1)),
        md.strip(),
    )
    return out
