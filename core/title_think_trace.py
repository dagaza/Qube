"""
Think-trace analysis for title-generation evaluation (diagnostic only).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.sidecar_prompts import (
    _TITLE_WORD_RE,
    _prepare_title_candidate,
    _quoted_topic_from_user_prompt,
)

_THINK_BLOCK_RE = re.compile(
    r"(?is)<(?:redacted_)?think(?:ing)?>\s*(.*?)(?:</(?:redacted_)?think(?:ing)?>|$)"
)
_THINK_OPEN_RE = re.compile(r"(?is)<(?:redacted_)?think(?:ing)?>")
_TITLE_CASE_PHRASE_RE = re.compile(
    r"\b([A-Z][\w']+(?:\s+(?:[A-Z][\w']+|of|the|and|in|for|to|[A-Z]{2,}))+)\b"
)


@dataclass
class ThinkTraceAnalysis:
    raw_completion: str = ""
    think_content: str = ""
    answer_content: str = ""
    had_think_block: bool = False
    candidate_in_reasoning: bool = False
    reasoning_candidate: str = ""
    answer_candidate: str = ""
    best_known_candidate: str = ""
    reasoning_has_best_title: bool = False
    answer_has_best_title: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_think_and_answer(raw: str) -> tuple[str, str, bool]:
    text = raw or ""
    match = _THINK_BLOCK_RE.search(text)
    if not match:
        return "", text.strip(), bool(_THINK_OPEN_RE.search(text))
    think = match.group(1).strip()
    answer = _THINK_BLOCK_RE.sub("", text).strip()
    return think, answer, True


def _title_like_phrases(text: str, *, max_words: int = 6) -> list[str]:
    phrases: list[str] = []
    for match in _TITLE_CASE_PHRASE_RE.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" \"'.,!?;:")
        words = _TITLE_WORD_RE.findall(phrase)
        if 2 <= len(words) <= max_words:
            polished = _prepare_title_candidate(
                " ".join(words[:max_words]),
                user_prompt="",
                assistant_reply="",
            )
            if polished and polished not in phrases:
                phrases.append(polished)
    return phrases


def analyze_think_trace(
    raw: str,
    *,
    user_prompt: str = "",
    assistant_reply: str = "",
    final_title: str = "",
) -> ThinkTraceAnalysis:
    think, answer, had_think = split_think_and_answer(raw)
    reasoning_phrases = _title_like_phrases(think)
    answer_phrases = _title_like_phrases(answer)

    quoted = _quoted_topic_from_user_prompt(user_prompt)
    best = (final_title or "").strip()
    if not best and quoted:
        best = quoted

    reasoning_candidate = reasoning_phrases[0] if reasoning_phrases else ""
    answer_candidate = answer_phrases[0] if answer_phrases else ""

    reasoning_has_best = bool(
        best
        and reasoning_candidate
        and best.lower() in think.lower()
    ) or (
        bool(best)
        and best.lower() in think.lower()
        and len(best.split()) >= 2
    )
    answer_has_best = bool(best and best.lower() in answer.lower())

    candidate_in_reasoning = bool(reasoning_candidate) or reasoning_has_best

    return ThinkTraceAnalysis(
        raw_completion=raw or "",
        think_content=think,
        answer_content=answer,
        had_think_block=had_think,
        candidate_in_reasoning=candidate_in_reasoning,
        reasoning_candidate=reasoning_candidate,
        answer_candidate=answer_candidate,
        best_known_candidate=best,
        reasoning_has_best_title=reasoning_has_best,
        answer_has_best_title=answer_has_best,
    )


def aggregate_think_trace_metrics(analyses: list[ThinkTraceAnalysis]) -> dict[str, Any]:
    n = len(analyses) or 1
    return {
        "count": len(analyses),
        "pct_had_think_block": 100.0 * sum(1 for a in analyses if a.had_think_block) / n,
        "pct_candidate_in_reasoning": 100.0
        * sum(1 for a in analyses if a.candidate_in_reasoning)
        / n,
        "pct_reasoning_has_best_title": 100.0
        * sum(1 for a in analyses if a.reasoning_has_best_title)
        / n,
        "pct_answer_has_best_title": 100.0
        * sum(1 for a in analyses if a.answer_has_best_title)
        / n,
        "pct_reasoning_nonempty": 100.0
        * sum(1 for a in analyses if (a.think_content or "").strip())
        / n,
        "pct_answer_nonempty": 100.0
        * sum(1 for a in analyses if (a.answer_content or "").strip())
        / n,
    }


__all__ = [
    "ThinkTraceAnalysis",
    "aggregate_think_trace_metrics",
    "analyze_think_trace",
    "split_think_and_answer",
]
