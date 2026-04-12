"""
Native streaming path only: optional removal of brief meta-instruction prefaces
(\"Provide a concise…\", \"We should…\") at the **start** of the assistant reply.

Conservative: fast path when the reply clearly does not begin with meta phrasing;
otherwise buffers briefly until a sentence break or a small cap, then applies one strip.

Does not replace system-prompt shaping; use together with stronger internal instructions.
"""
from __future__ import annotations

import re

# If the buffer matches this, we may be in a meta preface — wait for a natural break or cap.
_META_LEAD = re.compile(
    r"(?is)^\s*(?:"
    r"provide\s+(?:a|an|the)?\s*(?:concise\s+|clear\s+|brief\s+|direct\s+)?(?:answer|explanation|response|summary|overview)\b|"
    r"provide\s+(?:a|an|the)?\s+direct\b|"
    r"we\s+(?:should|need\s+to|must|will)\s+|"
    r"let\'?s\s+(?:start|begin|answer|break\s+this\s+down|look\s+at|provide)\b|"
    r"to\s+(?:answer|address|respond\s+to)\s+(?:this|the|that)\b|"
    r"here(?:'s| is)\s+(?:a|an|the|my)\s+(?:concise\s+)?(?:answer|summary|response|explanation)\b|"
    r"(?:the\s+)?(?:task|goal)\s+is\s+to\b|"
    r"i\s+(?:will|'?ll|am\s+going\s+to)\s+(?:answer|explain|provide|give)\b"
    r")"
)

# Applied at most once to the leading buffered segment.
_STRIP_ONCE = re.compile(
    r"(?is)^\s*(?:"
    r"provide\s+(?:a|an|the)?\s*(?:concise\s+|clear\s+|brief\s+|direct\s+)?(?:answer|explanation|response|summary|overview)\b"
    r"[^.!?\n]{0,220}[.!?:\-–—]?\s*"
    r"|provide\s+(?:a|an|the)?\s+direct\b[^.!?\n]{0,120}[.!?:\-–—]?\s*"
    r"|we\s+(?:should|need\s+to|must|will)\s+[^.!?\n]{0,260}[.!?:]\s*"
    r"|let\'?s\s+(?:start|begin|answer|break\s+this\s+down|look\s+at|provide)\b[^.!?\n]{0,200}[.!?:]\s*"
    r"|to\s+(?:answer|address|respond\s+to)\s+(?:this|the|that)\s+[^.!?\n]{0,120}[.!?:]\s*"
    r"|here(?:'s| is)\s+(?:a|an|the|my)\s+(?:concise\s+)?(?:answer|summary|response|explanation)\b[^.!?\n]{0,200}[.!?:]\s*"
    r"|(?:the\s+)?(?:task|goal)\s+is\s+to\b[^.!?\n]{0,200}[.!?:]\s*"
    r"|i\s+(?:will|'?ll|am\s+going\s+to)\s+(?:answer|explain|provide|give)\b[^.!?\n]{0,160}[.!?:]\s*"
    r")"
)

# Meta phrases we target almost always start with these letters (English).
_RISKY_FIRST = frozenset("pPwWlLtThHiI")


class LeadingMetaInstructionStripper:
    __slots__ = ("_buf", "_done")

    _max_hold = 360

    def __init__(self) -> None:
        self._buf = ""
        self._done = False

    def feed(self, chunk: str) -> str:
        if self._done:
            return chunk
        if not chunk:
            return ""
        self._buf += chunk
        return self._drain_if_ready()

    def flush(self) -> str:
        if self._done:
            return ""
        self._done = True
        out = _STRIP_ONCE.sub("", self._buf, count=1)
        self._buf = ""
        return out

    def _drain_if_ready(self) -> str:
        b = self._buf
        if not b:
            return ""

        t = b.lstrip()
        if not t:
            return ""

        # Fast path: common answer starters — never our meta patterns
        if t[0] not in _RISKY_FIRST and len(b) >= 1:
            self._done = True
            out = _STRIP_ONCE.sub("", b, count=1)
            self._buf = ""
            return out

        # Looks like a meta lead — wait for end of preface or cap
        if _META_LEAD.match(b):
            if _natural_break(b) or len(b) >= self._max_hold:
                self._done = True
                out = _STRIP_ONCE.sub("", b, count=1)
                self._buf = ""
                return out
            return ""

        # Risky first letter but not (yet) a meta lead — release after a short horizon
        if len(b) >= 14 and not _META_LEAD.match(b):
            self._done = True
            out = _STRIP_ONCE.sub("", b, count=1)
            self._buf = ""
            return out

        if len(b) >= self._max_hold:
            self._done = True
            out = _STRIP_ONCE.sub("", b, count=1)
            self._buf = ""
            return out
        return ""


def _natural_break(s: str) -> bool:
    if "\n\n" in s:
        return True
    return bool(re.search(r".{12,}[.!?](?:\s+|[\r\n])", s))
