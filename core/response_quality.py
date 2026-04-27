from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Optional

QualityConfidence = Literal["low", "medium", "high"]

_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")
_ABSOLUTE_RE = re.compile(r"\b(always|never|guaranteed|definitely|certainly)\b", re.I)
_GROUNDING_RE = re.compile(r"\b(maybe|likely|depends|in general|typically|often|can)\b", re.I)
_USELESS_RE = re.compile(
    r"\b(it depends|can't help|cannot help|not sure|i don't know|no idea)\b",
    re.I,
)


@dataclass
class ResponseQualityResult:
    score: float  # 0.0 - 1.0
    issues: list[str]
    confidence: QualityConfidence
    reasoning: Optional[str]


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def _last_user_query(user_query: str) -> str:
    t = (user_query or "").strip()
    if "USER QUERY:" in t:
        # Handle retrieval wrapper style payloads.
        t = t.rsplit("USER QUERY:", 1)[-1].strip()
    return t


def _query_is_complex(q: str) -> bool:
    low = q.lower()
    if len(q) > 120:
        return True
    return any(k in low for k in ("explain", "compare", "why", "how", "summarize", "steps", "analyze"))


def _requires_brief(q: str) -> bool:
    low = q.lower()
    return any(k in low for k in ("brief", "short", "concise", "one sentence"))


def _requires_list(q: str) -> bool:
    low = q.lower()
    return any(k in low for k in ("list", "bullet", "step-by-step", "steps"))


def _instruction_relevance_issue(query: str, output: str) -> bool:
    q_words = _token_set(query)
    o_words = _token_set(output)
    if not q_words or not o_words:
        return True
    overlap = len(q_words & o_words) / max(1, len(q_words))
    # Allow lower overlap for short/simple queries.
    min_overlap = 0.22 if _query_is_complex(query) else 0.15
    return overlap < min_overlap


def _completeness_issue(query: str, output: str) -> bool:
    t = (output or "").strip()
    if not t:
        return True
    if _query_is_complex(query) and len(t) < 80:
        return True
    if "?" in query and len(t) < 24:
        return True
    return False


def _coherence_issue(output: str) -> bool:
    low = (output or "").lower()
    # Lightweight contradiction patterns.
    contradiction_pairs = (
        ("yes", "no"),
        ("always", "never"),
        ("is", "is not"),
        ("can", "cannot"),
    )
    for a, b in contradiction_pairs:
        if a in low and b in low:
            return True
    # Broken discourse marker with no continuation.
    if re.search(r"\b(first|second|third)\b", low) and not re.search(r"\b(finally|in summary|therefore)\b", low):
        if len(low) < 60:
            return True
    return False


def _hallucination_signal_issue(output: str) -> bool:
    low = (output or "").lower()
    if _ABSOLUTE_RE.search(low) and not _GROUNDING_RE.search(low):
        # Flag strong certainty language unless softened.
        return True
    return False


def _utility_issue(query: str, output: str) -> bool:
    _ = query
    t = (output or "").strip()
    if len(t) < 12:
        return True
    if _USELESS_RE.search(t):
        # If it admits uncertainty but offers no next step, consider low utility.
        if not re.search(r"\b(try|check|verify|next|you can|step)\b", t, re.I):
            return True
    return False


def evaluate_response_quality(
    user_query: str,
    output: str,
    context: Optional[str] = None,
) -> ResponseQualityResult:
    _ = context
    query = _last_user_query(user_query)
    text = (output or "").strip()
    issues: list[str] = []
    reasoning_parts: list[str] = []
    score = 1.0

    if _instruction_relevance_issue(query, text):
        issues.append("low_relevance")
        reasoning_parts.append("weak lexical/topic overlap with user query")
        score -= 0.35

    if _requires_brief(query) and len(text) > 380:
        issues.append("constraint_missed_brief")
        reasoning_parts.append("query requested brief response but output is long")
        score -= 0.20

    if _requires_list(query) and not re.search(r"(^\s*[-*]\s+)|(^\s*\d+\.)", text, re.M):
        issues.append("constraint_missed_list")
        reasoning_parts.append("query requested list/steps but output is not structured as list")
        score -= 0.15

    if _completeness_issue(query, text):
        issues.append("incomplete_answer")
        reasoning_parts.append("response is too short for query complexity")
        score -= 0.20

    if _coherence_issue(text):
        issues.append("coherence_issue")
        reasoning_parts.append("internal contradiction or broken discourse cues detected")
        score -= 0.20

    if _hallucination_signal_issue(text):
        issues.append("overconfident_claim_signal")
        reasoning_parts.append("absolute certainty language without grounding cues")
        score -= 0.12

    if _utility_issue(query, text):
        issues.append("low_utility")
        reasoning_parts.append("response offers limited actionable help")
        score -= 0.18

    score = max(0.0, min(1.0, round(score, 4)))
    if score >= 0.75:
        conf: QualityConfidence = "high"
    elif score >= 0.45:
        conf = "medium"
    else:
        conf = "low"

    reasoning = "; ".join(reasoning_parts) if reasoning_parts else None
    return ResponseQualityResult(
        score=score,
        issues=issues,
        confidence=conf,
        reasoning=reasoning,
    )
