"""
Heuristic detection of GGUF ``tokenizer.chat_template`` strings that are unsafe for
end-user chat (channel / multi-phase scaffolding). Used only during prompt contract
resolution; does not add new formatting paths.
"""
from __future__ import annotations

import re

# Substrings / patterns (case-insensitive). Minimal set per product spec.
_SUBSTRING_MARKERS: tuple[tuple[str, str], ...] = (
    ("<|channel|>", r"<\|channel\|>"),
    ("<|message|>", r"<\|message\|>"),
    ("<|analysis|>", r"<\|analysis\|>"),
    ("<|final|>", r"<\|final\|>"),
    ("<|start|>", r"<\|start\|>"),
    ("<|end|>", r"<\|end\|>"),
)

# Assistant prefix anomaly (case-insensitive).
_ASSISTANT_ANOMALY = re.compile(r"<\|start\|>\s*assistant", re.IGNORECASE)

# Structured phase tags: <|...analysis...|> / <|...final...|> (not bare English words).
_PHASE_TAG = re.compile(
    r"<\|[^|]{0,48}(analysis|final)[^|]{0,48}\|>",
    re.IGNORECASE,
)


def is_unsafe_chat_template(template: str) -> tuple[bool, list[str]]:
    """
    Return (is_unsafe, reasons) for a tokenizer Jinja / chat template string.

    Strict, substring-based rules only; no model-name checks.
    """
    if not isinstance(template, str) or not template.strip():
        return False, []

    reasons: list[str] = []
    for label, pattern in _SUBSTRING_MARKERS:
        if re.search(pattern, template, flags=re.IGNORECASE):
            reasons.append(f"contains {label}")

    if _ASSISTANT_ANOMALY.search(template):
        reasons.append("contains <|start|>assistant")

    for m in _PHASE_TAG.finditer(template):
        tag = m.group(0).lower()
        r = f"structured phase tag: {tag}"
        if r not in reasons:
            reasons.append(r)

    seen: set[str] = set()
    uniq: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    return (bool(uniq), uniq)
