from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Literal

from core.harmony_protocol import is_harmony_contract
from core.prompt_contract import PromptContract

Severity = Literal["low", "medium", "high"]

_ADVISORY_THRESHOLD = 0.5
_HARD_THRESHOLD = 0.9
_CLUSTER_WINDOW = 40
_CLUSTER_MIN_COUNT = 4

_ENGLISH_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "from",
        "of",
        "as",
        "and",
        "or",
        "for",
        "with",
        "by",
        "during",
        "that",
        "this",
        "it",
        "is",
        "was",
        "were",
        "be",
        "are",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "not",
        "but",
        "if",
        "then",
        "than",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "what",
        "how",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "just",
        "also",
        "into",
        "over",
        "after",
        "before",
        "between",
        "through",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
    }
)

_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[INST\]|\[/INST\]", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),
    re.compile(r"<\|assistant\|>|</\|assistant\|>", re.I),
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
_STOP_ARTIFACT_END = re.compile(
    r"(?:<\|im_end\|>|</\|assistant\|>|\[/INST\]|\[INST\])\s*$",
    re.I,
)
_TOKENISH = re.compile(r"[a-zA-Z0-9]")
_WORD = re.compile(r"[a-zA-Z0-9]{2,}")
_STRUCTURED_BULLET_LINE = re.compile(
    r"^\s*[-*+]\s+\*\*.+?\*\*\s*[—\-–:]\s*.+",
    re.M,
)
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_BRACKET_LOOP = re.compile(r"(\[[^\]]+\])\1\1", re.I)


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


@dataclass(frozen=True)
class DegenerationAnalysis:
    score: float
    flagged: bool
    retry_eligible: bool
    top_offender: str | None
    top_count: int
    top_ratio: float
    clustered: bool


@dataclass
class OutputValidationResult:
    is_valid: bool
    issues: list[str]
    severity: Severity
    degeneration_score: float | None = None
    degeneration_retry_eligible: bool | None = None
    degeneration_top_offender: str | None = None
    degeneration_clustered: bool | None = None


def _content_word_count(ngram_words: list[str]) -> int:
    return sum(1 for w in ngram_words if w not in _ENGLISH_STOPWORDS and len(w) > 2)


def _repeat_count_threshold(word_count: int, list_heavy: bool) -> int:
    if list_heavy:
        if word_count < 500:
            return 6
        if word_count < 1500:
            return 8
        return 10
    if word_count < 500:
        return 4
    if word_count < 1500:
        return 6
    return 8


def _min_repeat_ratio(word_count: int) -> float:
    if word_count < 500:
        return 0.025
    if word_count < 1500:
        return 0.015
    return 0.008


def _is_clustered(
    positions: list[int],
    *,
    min_count: int = _CLUSTER_MIN_COUNT,
    max_span: int = _CLUSTER_WINDOW,
) -> bool:
    if len(positions) < min_count:
        return False
    sorted_pos = sorted(positions)
    for i in range(len(sorted_pos) - min_count + 1):
        if sorted_pos[i + min_count - 1] - sorted_pos[i] <= max_span:
            return True
    return False


def _score_offender(
    count: int,
    threshold: int,
    ratio: float,
    min_ratio: float,
    clustered: bool,
) -> float:
    if count < threshold or ratio < min_ratio:
        return 0.0
    count_norm = min(1.0, count / max(threshold * 2, threshold + 4))
    ratio_norm = min(1.0, ratio / max(min_ratio * 3, min_ratio + 0.01))
    base = 0.5 * count_norm + 0.5 * ratio_norm
    if clustered:
        return min(1.0, base + 0.35)
    return min(0.45, base * 0.65)


def _consecutive_word_run_score(words: list[str]) -> tuple[float, str | None, int, bool]:
    if len(words) < _CLUSTER_MIN_COUNT:
        return 0.0, None, 0, False
    run = 1
    best_run = 1
    best_word = words[0]
    for i in range(1, len(words)):
        if words[i] == words[i - 1] and len(words[i]) >= 2:
            run += 1
            if run > best_run:
                best_run = run
                best_word = words[i]
        else:
            run = 1
    if best_run >= 6:
        return 1.0, best_word, best_run, True
    if best_run >= _CLUSTER_MIN_COUNT:
        return 0.95, best_word, best_run, True
    return 0.0, None, 0, False


def _empty_degeneration_analysis() -> DegenerationAnalysis:
    return DegenerationAnalysis(
        score=0.0,
        flagged=False,
        retry_eligible=False,
        top_offender=None,
        top_count=0,
        top_ratio=0.0,
        clustered=False,
    )


def _finalize_degeneration(score: float, offender: str | None, count: int, ratio: float, clustered: bool) -> DegenerationAnalysis:
    return DegenerationAnalysis(
        score=round(score, 3),
        flagged=score >= _ADVISORY_THRESHOLD,
        retry_eligible=score >= _HARD_THRESHOLD,
        top_offender=offender,
        top_count=count,
        top_ratio=round(ratio, 4),
        clustered=clustered,
    )


def analyze_degeneration(text: str) -> DegenerationAnalysis:
    t = (text or "").strip()
    if not t:
        return _empty_degeneration_analysis()

    if _BRACKET_LOOP.search(t.lower()):
        return _finalize_degeneration(1.0, "[bracket_loop]", 3, 1.0, True)

    analysis = _prose_for_degeneration_check(t).lower()
    words = _WORD.findall(analysis)
    if len(words) < 8:
        return _empty_degeneration_analysis()

    list_heavy = _structured_list_or_table_heavy(t)
    word_count = len(words)
    threshold = _repeat_count_threshold(word_count, list_heavy)
    min_ratio = _min_repeat_ratio(word_count)
    chunk_sizes = (4,) if list_heavy else (2, 3, 4)

    best_score = 0.0
    best_offender: str | None = None
    best_count = 0
    best_ratio = 0.0
    best_clustered = False

    run_score, run_word, run_count, run_clustered = _consecutive_word_run_score(words)
    if run_score > best_score:
        best_score = run_score
        best_offender = run_word
        best_count = run_count
        best_ratio = run_count / max(word_count, 1)
        best_clustered = run_clustered

    for size in chunk_sizes:
        total_ngrams = max(1, len(words) - size + 1)
        positions_by_ngram: dict[str, list[int]] = defaultdict(list)
        for i in range(total_ngrams):
            ngram_words = words[i : i + size]
            if _content_word_count(ngram_words) == 0:
                continue
            ngram = " ".join(ngram_words)
            positions_by_ngram[ngram].append(i)

        for ngram, positions in positions_by_ngram.items():
            count = len(positions)
            ratio = count / total_ngrams
            clustered = _is_clustered(positions)
            subscore = _score_offender(count, threshold, ratio, min_ratio, clustered)
            if subscore > best_score:
                best_score = subscore
                best_offender = ngram
                best_count = count
                best_ratio = ratio
                best_clustered = clustered

    if best_score <= 0.0:
        return _empty_degeneration_analysis()

    return _finalize_degeneration(
        best_score,
        best_offender,
        best_count,
        best_ratio,
        best_clustered,
    )


def _degeneration(text: str) -> bool:
    return analyze_degeneration(text).flagged


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
    degeneration_score: float | None = None
    degeneration_retry_eligible: bool | None = None

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

    degeneration = analyze_degeneration(text)
    if degeneration.flagged:
        issues.append("degeneration")
        degeneration_score = degeneration.score
        degeneration_retry_eligible = degeneration.retry_eligible
        if degeneration.retry_eligible:
            severity = "high"
        elif severity != "high":
            severity = "medium"

    return OutputValidationResult(
        is_valid=not issues,
        issues=issues,
        severity=severity,
        degeneration_score=degeneration_score,
        degeneration_retry_eligible=degeneration_retry_eligible,
        degeneration_top_offender=degeneration.top_offender if degeneration.flagged else None,
        degeneration_clustered=degeneration.clustered if degeneration.flagged else None,
    )
