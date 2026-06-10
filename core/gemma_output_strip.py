"""
Gemma 4 output cleanup for the native stream path.

Gemma can echo internal system-alignment instructions, then emit the real answer
inside ``<|channel>thought`` (sometimes with malformed ``<channel|>`` fragments).
"""
from __future__ import annotations

import re

# Mirrors _INTERNAL_ALIGN_DEFAULT / partial echoes seen in production logs.
_INTERNAL_ALIGN_ECHO = re.compile(
    r"(?is)^\s*"
    r"(?:start directly with the answer content in natural language\.?\s*)?"
    r"(?:do not include preamble, planning, or meta commentary\.?\s*)?"
    r"(?:do not restate or analyze the user'?s request\.?\s*)?"
    r"(?:write only what the user should see\.?\s*)?"
    r"(?:keep the response natural and focused\.?\s*)+"
)

_GEMMA_THOUGHT_MARKER = re.compile(r"(?is)<\|channel>thought\b")

_GEMMA_THOUGHT_BODY = re.compile(
    r"(?is)"
    r"^.*?<\|channel>thought\s*\n?"
    r"(?:<\|?channel\|?>)?"
    r"(.+)$"
)

_GEMMA_CONTROL_INLINE = re.compile(
    r"(?is)<\|?channel\|?>|<\|think\|>|</?\|think\|>"
)

_GEMMA_CONTROL_FRAGMENT = re.compile(
    r"(?is)<\|channel>thought\b[^<\n]*"
)

_ALIGN_ECHO_START = re.compile(
    r"(?is)^\s*(?:do not include preamble|start directly with the answer)"
)


def is_gemma_model_identity(*, model_name: str = "", model_path: str = "") -> bool:
    blob = f"{model_name} {model_path}".lower()
    return "gemma" in blob


def _looks_gemma_artifact(text: str) -> bool:
    low = (text or "").lower()
    if _GEMMA_THOUGHT_MARKER.search(low):
        return True
    if "<channel|>" in low or "<|channel|>" in low:
        return True
    return bool(
        _INTERNAL_ALIGN_ECHO.match((text or "").strip())
        and "do not include preamble" in low
    )


def strip_gemma_output_artifacts(text: str) -> str:
    """Remove Gemma thought-channel scaffolding and leading instruction echoes."""
    if not text or not text.strip():
        return text
    if not _looks_gemma_artifact(text):
        t = (text or "").strip()
        if _INTERNAL_ALIGN_ECHO.match(t) and len(t) < 280:
            return ""
        return text

    m = _GEMMA_THOUGHT_BODY.match((text or "").strip())
    if m:
        body = _GEMMA_CONTROL_INLINE.sub("", m.group(1)).strip()
        if len(body) >= 16:
            return body

    t = _INTERNAL_ALIGN_ECHO.sub("", text, count=1)
    t = _GEMMA_THOUGHT_MARKER.sub("", t)
    t = _GEMMA_CONTROL_INLINE.sub("", t)
    t = _GEMMA_CONTROL_FRAGMENT.sub("", t)
    t = t.strip()
    if t and _INTERNAL_ALIGN_ECHO.match(t) and len(t) < 280:
        return ""
    return t


def _longest_suffix_prefix(s: str, needle: str) -> int:
    max_l = min(len(s), len(needle) - 1)
    for length in range(max_l, 0, -1):
        if needle.lower().startswith(s[-length:].lower()):
            return length
    return 0


def _clean_controls(fragment: str) -> str:
    if not fragment:
        return ""
    return _GEMMA_CONTROL_INLINE.sub("", fragment)


class GemmaThoughtStreamFilter:
    """
    Swallow instruction echoes before ``<|channel>thought``; emit thought-body text.

    Token-chunk safe; if no thought marker appears, passthrough after a short horizon
    when the stream does not look like a control-token prefix.
    """

    __slots__ = ("_buf", "_phase", "_done")

    _MARKER = "<|channel>thought"
    _PRE_HOLD_MAX = 400

    def __init__(self) -> None:
        self._buf = ""
        self._phase = "pre"
        self._done = False

    def feed(self, chunk: str) -> str:
        if self._done:
            return _clean_controls(chunk)
        if not chunk:
            return ""
        self._buf += chunk
        return self._drain()

    def flush(self) -> str:
        if self._done:
            return ""
        self._done = True
        rest = strip_gemma_output_artifacts(self._buf)
        self._buf = ""
        return rest

    def _drain(self) -> str:
        if self._phase == "passthrough":
            if not self._buf:
                return ""
            out = self._buf
            self._buf = ""
            return _clean_controls(out)

        if self._phase == "pre":
            low = self._buf.lower()
            idx = low.find(self._MARKER)
            if idx >= 0:
                rest = self._buf[idx + len(self._MARKER) :]
                rest = re.sub(r"(?is)^\s*(?:<\|?channel\|?>)?", "", rest, count=1)
                self._buf = rest
                self._phase = "thought"
                return self._drain()

            hold = _longest_suffix_prefix(self._buf, self._MARKER)
            if hold:
                self._buf = self._buf[-hold:]
                return ""

            stripped = self._buf.strip()
            if stripped and "<" not in self._buf:
                if _ALIGN_ECHO_START.match(stripped) or _INTERNAL_ALIGN_ECHO.match(
                    stripped
                ):
                    if len(self._buf) >= self._PRE_HOLD_MAX:
                        self._buf = ""
                    return ""
                self._phase = "passthrough"
                return self._drain()

            if "<" in self._buf and len(self._buf) >= self._PRE_HOLD_MAX:
                if _INTERNAL_ALIGN_ECHO.match(stripped):
                    self._buf = ""
                    return ""
            return ""

        # thought phase
        cleaned = _GEMMA_CONTROL_INLINE.sub("", self._buf)
        cleaned = _GEMMA_THOUGHT_MARKER.sub("", cleaned)
        self._buf = ""
        self._phase = "passthrough"
        return cleaned
