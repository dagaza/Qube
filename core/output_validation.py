from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from core.prompt_contract import PromptContract

Severity = Literal["low", "medium", "high"]

_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[INST\]|\[/INST\]", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),
    re.compile(r"^\s*(User|Assistant)\s*:", re.I | re.M),
    # Harmony / OSS chat template scaffolding leaked into completion text
    re.compile(r"<\|channel\|>|<\|message\|>|<\|final\|>", re.I),
    re.compile(
        r"<\|end\|>\s*<\|start\|>\s*assistant\s*<\|channel\|>\s*final\s*<\|message\|>",
        re.I,
    ),
    re.compile(r"<\|start\|>\s*assistant", re.I),
)
_ROLE_START = re.compile(r"^\s*(User|System)\s*:", re.I)
_ROLE_DIALOG = re.compile(r"(?:^|\n)\s*User\s*:.*(?:^|\n)\s*Assistant\s*:", re.I | re.S)
_ABRUPT_END = re.compile(r"(?:\.\.\.|[,;:\-\(\[])\s*$")
_STOP_ARTIFACT_END = re.compile(r"(?:<\|im_end\|>|\[/INST\]|\[INST\])\s*$", re.I)
_TOKENISH = re.compile(r"[a-zA-Z0-9]")
_WORD = re.compile(r"[a-zA-Z0-9]{2,}")


@dataclass
class OutputValidationResult:
    is_valid: bool
    issues: list[str]
    severity: Severity


def _template_leakage(text: str) -> bool:
    for pat in _LEAK_PATTERNS:
        if pat.search(text):
            return True
    return False


def _role_confusion(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _ROLE_START.search(t):
        return True
    # Detect self-dialogue style output that should not happen in single assistant reply.
    return bool(_ROLE_DIALOG.search(t))


def _truncated_output(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 10 or not _TOKENISH.search(t):
        return True
    if _STOP_ARTIFACT_END.search(t):
        return True
    if _ABRUPT_END.search(t):
        return True
    return False


def _meta_preamble_only(text: str) -> bool:
    """Single-line bracketed planning / meta reply, not numeric citations like [1]."""
    t = (text or "").strip()
    if not t or "\n" in t:
        return False
    m = re.match(r"^\s*\[([^\]]+)\]\s*\.?\s*$", t)
    if not m:
        return False
    inner = (m.group(1) or "").strip()
    if len(inner) < 10:
        return False
    if inner.isdigit():
        return False
    return True


def _degeneration(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    # obvious token loops
    if re.search(r"(\[[^\]]+\])\1\1", low):
        return True
    words = _WORD.findall(low)
    if len(words) < 8:
        return False
    # repeated short phrase
    for size in (2, 3, 4):
        chunks = [" ".join(words[i : i + size]) for i in range(0, max(0, len(words) - size + 1))]
        if not chunks:
            continue
        freq: dict[str, int] = {}
        for c in chunks:
            freq[c] = freq.get(c, 0) + 1
            if freq[c] >= 4:
                return True
    return False


def validate_output(text: str, contract: PromptContract) -> OutputValidationResult:
    _ = contract
    issues: list[str] = []
    severity: Severity = "low"

    if _template_leakage(text):
        issues.append("template_leakage")
        severity = "high"

    if _meta_preamble_only(text):
        issues.append("meta_preamble")
        severity = "high"

    if _role_confusion(text):
        if "role_confusion" not in issues:
            issues.append("role_confusion")
        severity = "high"

    if _truncated_output(text):
        issues.append("truncated_output")
        if severity != "high":
            severity = "medium"

    if _degeneration(text):
        issues.append("degeneration")
        if severity != "high":
            severity = "medium"

    return OutputValidationResult(
        is_valid=not issues,
        issues=issues,
        severity=severity,
    )
