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
    """
    if not text:
        return text

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
    return normalize_combined_numeric_citations(t)
