"""
Streaming strip of <redacted_thinking>...</redacted_thinking> blocks (e.g. DeepSeek-R1 style).
Used only on the internal llama.cpp path; external servers often strip these server-side.
"""
from __future__ import annotations

_OPEN = "<redacted_thinking>"
_CLOSE = "</redacted_thinking>"


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

    __slots__ = ("_thinking", "_buf")

    def __init__(self) -> None:
        self._thinking = False
        self._buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out_parts: list[str] = []
        while True:
            if not self._thinking:
                idx = self._buf.find(_OPEN)
                if idx >= 0:
                    out_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(_OPEN) :]
                    self._thinking = True
                    continue
                hold = _longest_suffix_that_is_prefix_of(self._buf, _OPEN)
                if hold:
                    out_parts.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out_parts.append(self._buf)
                    self._buf = ""
                return "".join(out_parts)

            idx = self._buf.find(_CLOSE)
            if idx >= 0:
                self._buf = self._buf[idx + len(_CLOSE) :]
                self._thinking = False
                continue

            hold = _longest_suffix_that_is_prefix_of(self._buf, _CLOSE)
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
            return ""
        tail = self._buf
        self._buf = ""
        return tail
