"""Remove Harmony / OSS-style template scaffolding from model text (runtime output)."""
from __future__ import annotations

import re

# Log-derived bridge: <|end|><|start|>assistant<|channel|>final<|message|>
_HARMONY_BRIDGE = re.compile(
    r"(?is)<\|end\|>\s*<\|start\|>\s*assistant\s*<\|channel\|>\s*final\s*<\|message\|>",
)
# Instruction-echo preface often preceding the bridge (one shot, non-greedy).
_INSTRUCTION_ECHO = re.compile(
    r"(?is)We\s+need\s+to\s+explain.*?Provide\s+concise\.?\s*",
)
# Untagged OSS/Harmony scratchpad tail observed after a partial final answer.
_SCRATCHPAD_TAIL = re.compile(
    r"(?is)\s*(?:[?.!…]{3,}\s*)?(?:"
    r"We\s+need\s+to\s+(?:answer|explain)|"
    r"We\s+should\s+produce|"
    r"We\s+have\s+sources?|"
    r"Source\s+\d+\s+(?:indicates|says)|"
    r"Let's\s+produce\s+answer"
    r")\b.*$",
)
# Keep the natural answer if a stop/scratchpad cut leaves punctuation noise at the end.
_TRAILING_NOISE = re.compile(r"(?s)(?:\s*[?.!…]){3,}\s*$")
_META_COMMAND_PREFIX = re.compile(
    r"(?is)^\s*provide\s+final\s+answer\b\s*(?:[.!?:\-–—]+|\n+|\s+)?"
)
# Any remaining <|...|> control tokens (bounded label).
_CONTROL_TOKEN = re.compile(r"<\|[^|\n]{1,56}\|>")


def strip_harmony_oss_artifacts(text: str) -> str:
    if not text or not text.strip():
        return text
    t = _INSTRUCTION_ECHO.sub("", text, count=1)
    t = _HARMONY_BRIDGE.sub("", t)
    t = _CONTROL_TOKEN.sub("", t)
    t = _META_COMMAND_PREFIX.sub("", t, count=1)
    t = _SCRATCHPAD_TAIL.sub("", t, count=1)
    t = _TRAILING_NOISE.sub("", t)
    return t
