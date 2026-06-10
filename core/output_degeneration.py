"""
Model-family-agnostic output degeneration detector.

Scores repetition, malformed lists, self-corrections, meta commentary,
unfinished bullets, punctuation loops, markdown explosions, and token
entropy collapse. High composite scores mark turns unreliable so they
are not replayed into future context.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional

from core.harmony_degeneration import (
    find_degeneration_start,
    harmony_tail_degenerate,
    is_harmony_orphan_stream_fragment,
)

OutputDegenerationRisk = Literal["LOW", "MEDIUM", "HIGH"]

HISTORY_SUPPRESSION_PLACEHOLDER = (
    "[Previous assistant response suppressed due to degeneration detection]"
)

HIGH_THRESHOLD = 0.55
MEDIUM_THRESHOLD = 0.35

_COMPOSITE_WEIGHTS: dict[str, float] = {
    "repetition": 0.35,
    "malformed_list": 0.20,
    "meta_commentary": 0.20,
    "punctuation_loop": 0.15,
    "truncation": 0.10,
}

_SELF_CORRECTION = re.compile(
    r"(?i)(?:^|\n)\s*(?:"
    r"wait,?\s*(?:actually|no)|"
    r"actually,?\s*i\s+(?:meant|should)|"
    r"correction:|"
    r"let me (?:correct|rephrase|try again)|"
    r"on second thought"
    r")\b"
)
_META_COMMENTARY = re.compile(
    r"(?i)(?:^|\n)\s*(?:"
    r"sorry|just kidding|i mean|my apologies|"
    r"oops|never mind|scratch that|"
    r"as an ai|as a language model|"
    r"the user (?:asked|wants|said|is asking)"
    r")\b"
)
_PLANNING_VOICE = re.compile(
    r"(?i)^\s*(?:we need to|we should|let'?s clarify|"
    r"i will (?:now|first)|step \d+:|"
    r"provide final answer|final channel)"
)
_UNFINISHED_BOLD_ITEM = re.compile(r"\n\d+\.\s+\*\*[^\n*]+$")
_UNFINISHED_BULLET = re.compile(r"\n[-*]\s+\S[^\n]{0,200}$")
_TRAILING_ORPHAN_NUMBER = re.compile(r"\n\s*\d+\.\s*$")
_BROKEN_LIST_LINE = re.compile(
    r"(?i)\n\s*\d+\.\s*\*\*[^\n]*(?:\?\?|\*\*\s*‑\s*\d+\s*–\s*–|‑\s*\d+\s*–\s*–)"
)
_BARE_NUMBERED_RUN = re.compile(r"(?:^|\n)\s*\d+\.\s*(?:\n|$)", re.M)
_PUNCT_RUN = re.compile(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{24,}")
_MARKDOWN_HEADER_SPAM = re.compile(r"(?:^|\n)\s*#{2,6}\s+\S", re.M)
_BOLD_SPAM = re.compile(r"\*\*")
_ORPHAN_BOLD = re.compile(r"\*\*[^*\n]{0,120}(?:\?\?|‑‑|\?\s*$)")


@dataclass(frozen=True)
class OutputDegenerationComponents:
    repetition: float
    malformed_list: float
    self_correction: float
    meta_commentary: float
    unfinished_bullet: float
    punctuation_loop: float
    markdown_explosion: float
    entropy_collapse: float
    truncation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "repetition": round(self.repetition, 3),
            "malformed_list": round(self.malformed_list, 3),
            "self_correction": round(self.self_correction, 3),
            "meta_commentary": round(self.meta_commentary, 3),
            "unfinished_bullet": round(self.unfinished_bullet, 3),
            "punctuation_loop": round(self.punctuation_loop, 3),
            "markdown_explosion": round(self.markdown_explosion, 3),
            "entropy_collapse": round(self.entropy_collapse, 3),
            "truncation": round(self.truncation, 3),
        }


@dataclass(frozen=True)
class OutputDegenerationResult:
    composite_score: float
    risk: OutputDegenerationRisk
    components: OutputDegenerationComponents
    flags: tuple[str, ...]

    @property
    def should_mark_unreliable(self) -> bool:
        return self.risk == "HIGH"

    def trace_fields(self) -> dict[str, object]:
        return {
            "output_degeneration_score": round(self.composite_score, 3),
            "output_degeneration_risk": self.risk,
            "output_degeneration_flags": list(self.flags),
            "output_degeneration_components": self.components.as_dict(),
            "output_degeneration_unreliable": self.should_mark_unreliable,
        }


def should_mark_turn_unreliable(result: OutputDegenerationResult) -> bool:
    """True when the turn must not be trusted in future session history."""
    return result.should_mark_unreliable


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_repetition(text: str) -> float:
    t = (text or "").strip()
    if len(t) < 24:
        return 0.0
    tail = t[-600:]
    atoms = tail.split()
    if len(atoms) >= 8:
        run = 1
        best = 1
        for i in range(1, len(atoms)):
            if atoms[i] == atoms[i - 1] and len(atoms[i]) <= 16:
                run += 1
                best = max(best, run)
            else:
                run = 1
        if best >= 6:
            return 1.0
        if best >= 4:
            return 0.75
        if best >= 3:
            return 0.45

    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if len(lines) >= 3:
        counts = Counter(lines)
        most_common_n, freq = counts.most_common(1)[0]
        if freq >= 3 and len(most_common_n) <= 48:
            return 0.85

    words = re.findall(r"[a-zA-Z]{3,}", tail.lower())
    if len(words) >= 12:
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        tri_counts = Counter(trigrams)
        top_n, top_f = tri_counts.most_common(1)[0]
        if top_f >= 4 and len(top_n) <= 40:
            return 0.7
    return 0.0


def _score_malformed_list(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    score = 0.0
    if find_degeneration_start(t) is not None:
        score = max(score, 1.0)
    if _BROKEN_LIST_LINE.search(t):
        score = max(score, 0.95)
    bare = _BARE_NUMBERED_RUN.findall(t[-400:])
    if len(bare) >= 4:
        score = max(score, 0.85)
    if _UNFINISHED_BOLD_ITEM.search(t):
        score = max(score, 0.7)
    return _clamp01(score)


def _score_self_correction(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    hits = len(_SELF_CORRECTION.findall(t))
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.55
    return 0.0


def _score_meta_commentary(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    score = 0.0
    if _META_COMMENTARY.search(t):
        score = max(score, 0.65)
    if _PLANNING_VOICE.search(t):
        score = max(score, 0.75)
    if score and _SELF_CORRECTION.search(t):
        score = max(score, 0.85)
    return _clamp01(score)


def _score_unfinished_bullet(text: str) -> float:
    t = (text or "").rstrip()
    if not t:
        return 0.0
    if _UNFINISHED_BULLET.search(t):
        return 0.8
    if _TRAILING_ORPHAN_NUMBER.search(t):
        return 0.85
    if is_harmony_orphan_stream_fragment(t[-24:]):
        return 0.75
    if re.search(r"\n\d+\.\s+\*\*[^*\n]{0,80}$", t[-120:]):
        return 0.7
    return 0.0


def _score_punctuation_loop(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    if harmony_tail_degenerate(t):
        return 1.0
    tail = t[-500:]
    if _PUNCT_RUN.search(tail):
        return 0.95
    spaced = re.search(
        r"[\s\u00a0\u202f\u2009\u200a\u2028.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{28,}",
        tail,
    )
    if spaced is not None:
        punct_chars = len(re.findall(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]", spaced.group(0)))
        if punct_chars >= 12:
            return 0.9
    return 0.0


def _score_markdown_explosion(text: str) -> float:
    t = (text or "").strip()
    if len(t) < 40:
        return 0.0
    score = 0.0
    headers = len(_MARKDOWN_HEADER_SPAM.findall(t))
    if headers >= 6:
        score = max(score, 0.85)
    elif headers >= 4:
        score = max(score, 0.55)
    bold_count = len(_BOLD_SPAM.findall(t))
    if bold_count >= 24:
        score = max(score, 0.9)
    elif bold_count >= 16:
        score = max(score, 0.6)
    if _ORPHAN_BOLD.search(t):
        score = max(score, 0.8)
    ratio = bold_count / max(len(t), 1)
    if ratio > 0.08:
        score = max(score, 0.7)
    return _clamp01(score)


def _score_entropy_collapse(text: str) -> float:
    """
    Detect character-level collapse (punctuation loops, symbol spam).

    Avoids false positives on normal prose with large numbers or commas.
    """
    t = (text or "").strip()
    if len(t) < 120:
        return 0.0
    tail = t[-400:]
    if _PUNCT_RUN.search(tail):
        return 0.95
    spaced = re.search(
        r"[\s\u00a0\u202f\u2009\u200a\u2028.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{28,}",
        tail,
    )
    if spaced is not None:
        punct_chars = len(re.findall(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]", spaced.group(0)))
        if punct_chars >= 12:
            return 0.9

    letters = sum(1 for c in tail if c.isalpha())
    if letters / max(len(tail), 1) < 0.25:
        return 0.85

    atoms = tail.split()
    if len(atoms) >= 10:
        run = 1
        best = 1
        for i in range(1, len(atoms)):
            if atoms[i] == atoms[i - 1] and len(atoms[i]) <= 3:
                run += 1
                best = max(best, run)
            else:
                run = 1
        if best >= 8:
            return 0.9

    return 0.0


def _score_truncation(text: str) -> float:
    t = (text or "").rstrip()
    if not t:
        return 0.0
    if t.endswith(("…", "...", "\u2026")) and len(t) >= 40:
        return 0.75
    if len(t) >= 40 and re.search(r"[,;:\(\[\-]\s*$", t):
        return 0.65
    if _score_unfinished_bullet(t) >= 0.7:
        return 0.7
    # Mid-sentence cut (no terminal punctuation) only when clearly incomplete.
    if (
        len(t) >= 80
        and t[-1].isalnum()
        and not re.search(r"[.!?][\"')\]]*\s*$", t)
        and re.search(r"\b(and|or|the|a|an|of|to|for|with|in|on|at|by)\s*$", t, re.I)
    ):
        return 0.55
    return 0.0


def _composite_score(components: OutputDegenerationComponents) -> float:
    repetition = max(components.repetition, components.entropy_collapse * 0.85)
    malformed_list = max(
        components.malformed_list,
        components.unfinished_bullet,
        components.markdown_explosion * 0.75,
    )
    meta = max(components.meta_commentary, components.self_correction)
    return _clamp01(
        repetition * _COMPOSITE_WEIGHTS["repetition"]
        + malformed_list * _COMPOSITE_WEIGHTS["malformed_list"]
        + meta * _COMPOSITE_WEIGHTS["meta_commentary"]
        + components.punctuation_loop * _COMPOSITE_WEIGHTS["punctuation_loop"]
        + components.truncation * _COMPOSITE_WEIGHTS["truncation"]
    )


def _critical_degeneration(components: OutputDegenerationComponents) -> bool:
    """Any one severe marker is enough to mark the turn unreliable."""
    return (
        components.malformed_list >= 0.85
        or components.punctuation_loop >= 0.85
        or components.unfinished_bullet >= 0.75
        or components.repetition >= 0.75
        or components.entropy_collapse >= 0.90
        or (
            components.entropy_collapse >= 0.85
            and (components.repetition >= 0.45 or components.punctuation_loop >= 0.45)
        )
        or components.truncation >= 0.70
        or max(components.meta_commentary, components.self_correction) >= 0.85
    )


def _risk_tier(score: float, components: OutputDegenerationComponents) -> OutputDegenerationRisk:
    if score >= HIGH_THRESHOLD or _critical_degeneration(components):
        return "HIGH"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _collect_flags(components: OutputDegenerationComponents) -> tuple[str, ...]:
    flags: list[str] = []
    checks = (
        (components.repetition, "repetition"),
        (components.malformed_list, "malformed_list"),
        (components.self_correction, "self_correction"),
        (components.meta_commentary, "meta_commentary"),
        (components.unfinished_bullet, "unfinished_bullet"),
        (components.punctuation_loop, "punctuation_loop"),
        (components.markdown_explosion, "markdown_explosion"),
        (components.entropy_collapse, "entropy_collapse"),
        (components.truncation, "truncation"),
    )
    for value, name in checks:
        if value >= 0.45:
            flags.append(name)
    return tuple(dict.fromkeys(flags))


def detect_output_degeneration(text: str) -> OutputDegenerationResult:
    """Score assistant output for cross-model degeneration."""
    t = (text or "").strip()
    if not t:
        components = OutputDegenerationComponents(0, 0, 0, 0, 0, 0, 0, 0, 0)
        return OutputDegenerationResult(0.0, "LOW", components, ())

    components = OutputDegenerationComponents(
        repetition=_score_repetition(t),
        malformed_list=_score_malformed_list(t),
        self_correction=_score_self_correction(t),
        meta_commentary=_score_meta_commentary(t),
        unfinished_bullet=_score_unfinished_bullet(t),
        punctuation_loop=_score_punctuation_loop(t),
        markdown_explosion=_score_markdown_explosion(t),
        entropy_collapse=_score_entropy_collapse(t),
        truncation=_score_truncation(t),
    )
    composite = _composite_score(components)
    return OutputDegenerationResult(
        composite_score=composite,
        risk=_risk_tier(composite, components),
        components=components,
        flags=_collect_flags(components),
    )


def detect_stream_degeneration(text: str) -> OutputDegenerationResult:
    """
    Stricter streaming-only scoring: ignore entropy/truncation on partial text.

    Used to abort live generation without false positives on numeric prose or
    incomplete sentences still being written.
    """
    t = (text or "").strip()
    if not t:
        components = OutputDegenerationComponents(0, 0, 0, 0, 0, 0, 0, 0, 0)
        return OutputDegenerationResult(0.0, "LOW", components, ())

    components = OutputDegenerationComponents(
        repetition=_score_repetition(t),
        malformed_list=_score_malformed_list(t),
        self_correction=_score_self_correction(t),
        meta_commentary=_score_meta_commentary(t),
        unfinished_bullet=_score_unfinished_bullet(t),
        punctuation_loop=_score_punctuation_loop(t),
        markdown_explosion=_score_markdown_explosion(t),
        entropy_collapse=0.0,
        truncation=0.0,
    )
    composite = _composite_score(components)
    return OutputDegenerationResult(
        composite_score=composite,
        risk=_risk_tier(composite, components),
        components=components,
        flags=_collect_flags(components),
    )


class OutputDegenerationStreamObserver:
    """
    Incrementally rescores streamed output; trips once composite risk is HIGH.

    Observer-only: does not mutate visible tokens.
    """

    __slots__ = (
        "_buffer",
        "_rescore_every",
        "_min_buffer",
        "_last_score",
        "_tripped",
        "_trip_reason",
        "_last_result",
    )

    def __init__(self, *, rescore_every: int = 120, min_buffer: int = 200) -> None:
        self._buffer = ""
        self._rescore_every = max(80, int(rescore_every))
        self._min_buffer = max(160, int(min_buffer))
        self._last_score = 0.0
        self._tripped = False
        self._trip_reason: Optional[str] = None
        self._last_result: Optional[OutputDegenerationResult] = None

    def observe(self, delta: str) -> bool:
        """Feed streamed delta; return True when HIGH risk detected."""
        if self._tripped:
            return True
        if not delta:
            return False
        self._buffer += delta
        if len(self._buffer) < self._min_buffer:
            return False
        if len(self._buffer) < self._rescore_every:
            return False
        result = detect_stream_degeneration(self._buffer)
        self._last_result = result
        self._last_score = result.composite_score
        if result.risk == "HIGH":
            self._tripped = True
            self._trip_reason = (
                f"output degeneration HIGH ({result.composite_score:.2f}; "
                f"flags={','.join(result.flags)})"
            )
            return True
        if len(self._buffer) > 2400:
            self._buffer = self._buffer[-1800:]
        return False

    def final_result(self) -> Optional[OutputDegenerationResult]:
        if self._tripped:
            return self._last_result
        if self._buffer.strip():
            return detect_output_degeneration(self._buffer)
        return self._last_result

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> Optional[str]:
        return self._trip_reason

    @property
    def last_score(self) -> float:
        return self._last_score
