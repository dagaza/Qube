"""
Streaming strip of thinking blocks (e.g. DeepSeek-R1 / Nemotron / Qwen3 style).
Used on the internal llama.cpp stream path; ``strip_reasoning_blocks_from_text`` is
also used for complete sidecar completions (titling, judge, etc.).
"""
from __future__ import annotations

import re

_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<redacted_thinking>", "</redacted_thinking>"),
    ("<thinking>", "</thinking>"),
    ("<think>", "</think>"),
)
_OPEN_TAGS = tuple(p[0] for p in _TAG_PAIRS)
_CLOSE_BY_OPEN = {open_tag: close_tag for open_tag, close_tag in _TAG_PAIRS}

# Complete-string cleanup (sidecar / non-streaming paths). Case-insensitive for Qwen3 <Think>.
_THINK_BLOCK_RE = re.compile(
    r"(?is)<(?:redacted_)?think(?:ing)?>\s*.*?\s*</(?:redacted_)?think(?:ing)?>"
)
_UNCLOSED_THINK_RE = re.compile(r"(?is)<(?:redacted_)?think(?:ing)?>\s*.*$")
_ORPHAN_THINK_TAG_RE = re.compile(r"(?is)</?(?:redacted_)?think(?:ing)?>")
_PIPE_THINK_BLOCK_RE = re.compile(r"(?is)<\|think\|>\s*.*?\s*<\|/think\|>")


def strip_reasoning_blocks_from_text(text: str) -> str:
    """Remove thinking/reasoning blocks and stray tags from a full completion string."""
    if not text or not text.strip():
        return text
    t = text
    for _ in range(8):
        t2 = _THINK_BLOCK_RE.sub("", t)
        t2 = _PIPE_THINK_BLOCK_RE.sub("", t2)
        if t2 == t:
            break
        t = t2
    t = _UNCLOSED_THINK_RE.sub("", t)
    t = _ORPHAN_THINK_TAG_RE.sub("", t)
    return t


def _longest_suffix_that_is_prefix_of(s: str, needle: str) -> int:
    """Max L in (0, len(needle)) such that s[-L:] == needle[:L]."""
    max_l = min(len(s), len(needle) - 1)
    for L in range(max_l, 0, -1):
        if needle.startswith(s[-L:]):
            return L
    return 0


class RedactedThinkingStreamFilter:
    """
    Token-chunk safe: tags may be split across arbitrary stream boundaries.
    When not inside a block, only holds back a short suffix that might still
    complete the opening tag so normal text does not wait on a full buffer.
    """

    __slots__ = ("_thinking", "_buf", "_close_tag")

    def __init__(self) -> None:
        self._thinking = False
        self._buf = ""
        self._close_tag = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out_parts: list[str] = []
        while True:
            if not self._thinking:
                idx = -1
                open_tag = ""
                for candidate in _OPEN_TAGS:
                    cand_idx = self._buf.find(candidate)
                    if cand_idx >= 0 and (idx < 0 or cand_idx < idx):
                        idx = cand_idx
                        open_tag = candidate
                if idx >= 0:
                    out_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(open_tag) :]
                    self._thinking = True
                    self._close_tag = _CLOSE_BY_OPEN[open_tag]
                    continue
                hold = max(
                    _longest_suffix_that_is_prefix_of(self._buf, tag)
                    for tag in _OPEN_TAGS
                )
                if hold:
                    out_parts.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out_parts.append(self._buf)
                    self._buf = ""
                return "".join(out_parts)

            close_tag = self._close_tag
            idx = self._buf.find(close_tag)
            if idx >= 0:
                self._buf = self._buf[idx + len(close_tag) :]
                self._thinking = False
                self._close_tag = ""
                continue

            hold = _longest_suffix_that_is_prefix_of(self._buf, close_tag)
            if hold:
                self._buf = self._buf[-hold:]
            else:
                self._buf = ""
            return "".join(out_parts)

    def flush(self) -> str:
        """Call at stream end: emit any safe remainder; drop unclosed thinking tail."""
        if self._thinking:
            self._buf = ""
            self._thinking = False
            self._close_tag = ""
            return ""
        tail = self._buf
        self._buf = ""
        return tail
