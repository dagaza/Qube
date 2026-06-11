from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from core.harmony_protocol import is_harmony_contract
from core.prompt_contract import PromptContract

Severity = Literal["low", "medium", "high"]

_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[INST\]|\[/INST\]", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),
    re.compile(r"^\s*(User|Assistant)\s*:", re.I | re.M),
    # Harmony / OSS chat template scaffolding leaked into completion text
    re.compile(r"<\|channel\|>|<\|message\|>|<\|final\|>", re.I),
    # Malformed partial tokens when a mismatched formatter (e.g. ChatML on Gemma Jinja) drifts
    re.compile(r"<\|?channel\|?>|<\|?message\|?>|<\|?start\|?>|<\|?final\|?>", re.I),
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
_STRUCTURED_BULLET_LINE = re.compile(
    r"^\s*[-*+]\s+\*\*.+?\*\*\s*[—\-–:]\s*.+",
    re.M,
)
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _structured_list_or_table_heavy(text: str) -> bool:
    bullets = len(_STRUCTURED_BULLET_LINE.findall(text))
    table_rows = sum(
        1
        for line in text.splitlines()
        if line.strip().startswith("|") and not _TABLE_SEPARATOR.match(line.strip())
    )
    return bullets >= 6 or table_rows >= 6 or (bullets >= 3 and table_rows >= 3)


def _prose_for_degeneration_check(text: str) -> str:
    """Strip repeated markdown scaffolding before n-gram loop detection."""
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = re.match(r"^[-*+]\s+\*\*(.+?)\*\*\s*[—\-–:]\s*(.*)$", line)
        if bullet:
            parts.append(f"{bullet.group(1)} {bullet.group(2)}")
            continue
        if line.startswith("|"):
            cells = [
                cell.strip()
                for cell in line.split("|")
                if cell.strip() and not re.fullmatch(r":?-{3,}:?", cell.strip())
            ]
            parts.extend(cells)
            continue
        parts.append(line)
    return " ".join(parts)


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
    if not t or not _TOKENISH.search(t):
        return True
    if _STOP_ARTIFACT_END.search(t):
        return True
    if _ABRUPT_END.search(t):
        return True
    # Short complete replies ("Hello", "Yes", "OK") are valid, not truncation.
    if len(t) < 10:
        return False
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
    analysis = _prose_for_degeneration_check(t).lower()
    words = _WORD.findall(analysis)
    if len(words) < 8:
        return False
    list_heavy = _structured_list_or_table_heavy(t)
    chunk_sizes = (4,) if list_heavy else (2, 3, 4)
    repeat_threshold = 6 if list_heavy else 4
    for size in chunk_sizes:
        chunks = [" ".join(words[i : i + size]) for i in range(0, max(0, len(words) - size + 1))]
        if not chunks:
            continue
        freq: dict[str, int] = {}
        for c in chunks:
            freq[c] = freq.get(c, 0) + 1
            if freq[c] >= repeat_threshold:
                return True
    return False


_PLANNING_ONLY = re.compile(
    r"(?is)^\s*(?:we\s+need\s+to|we\s+should|we\s+have|let'?s\s+clarify|the\s+user\s+says)\b"
)


def _harmony_no_final_answer(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _PLANNING_ONLY.match(t) and len(t) < 400:
        return True
    return False


def validate_output(text: str, contract: PromptContract) -> OutputValidationResult:
    issues: list[str] = []
    severity: Severity = "low"
    harmony = is_harmony_contract(contract)

    if harmony and _harmony_no_final_answer(text):
        issues.append("harmony_no_final_answer")
        severity = "high"

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
