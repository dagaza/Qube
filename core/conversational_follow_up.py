"""Preserve short conversational follow-ups after format-fallback replace."""
from __future__ import annotations

import re

from core.output_artifact_strip import strip_harmony_oss_artifacts
from core.output_validation import validate_output
from core.prompt_contract import PromptContract

_DUMMY_CONTRACT = PromptContract(
    mode="messages",
    chat_format="chatml",
    prompt=None,
    messages=[{"role": "user", "content": "hi"}],
    stop=[],
    template_source="fallback",
    confidence="medium",
)

_LIVELY_FOLLOW_UP_START = re.compile(
    r"^(?:"
    r"would you(?:'re| like)|"
    r"want(?: to hear| another| me to)|"
    r"shall i|should i|can i|"
    r"(?:need|hear) another|"
    r"another one|"
    r"let me know|"
    r"feel free|"
    r"interested in|"
    r"curious about|"
    r"anything else|"
    r"want more|"
    r"care for|"
    r"ready for"
    r")\b",
    re.I,
)
_UNSAFE_FOLLOW_UP = re.compile(
    r"(?:"
    r"we need to|we should|we have sources?|source \d+|"
    r"provide (?:final )?answer|let's produce|\[INST\]|<\|"
    r")",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_lively_conversational_follow_up(sentence: str) -> bool:
    """True for short user-directed invitations/questions, not answer body."""
    s = (sentence or "").strip()
    if len(s) < 8 or len(s) > 140:
        return False
    if _UNSAFE_FOLLOW_UP.search(s):
        return False
    if _LIVELY_FOLLOW_UP_START.search(s):
        return True
    if s.endswith("?"):
        return bool(re.search(r"\b(you|your|another|more|else|that)\b", s, re.I))
    return False


def is_safe_follow_up(sentence: str) -> bool:
    """Reject template leakage, degeneration, and meta-instruction tails."""
    s = strip_harmony_oss_artifacts(sentence or "").strip()
    if not s or s != (sentence or "").strip():
        # Stripping removed unsafe scaffolding.
        if not s:
            return False
    if _UNSAFE_FOLLOW_UP.search(s):
        return False
    validation = validate_output(s, _DUMMY_CONTRACT)
    if not validation.is_valid:
        return False
    if not is_lively_conversational_follow_up(s):
        return False
    return True


def extract_follow_up_candidate(streamed: str) -> str | None:
    """Return the last lively follow-up sentence/paragraph from streamed text."""
    cleaned = strip_harmony_oss_artifacts(streamed or "").strip()
    if not cleaned:
        return None

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    if len(paragraphs) >= 2:
        tail = paragraphs[-1]
        if is_safe_follow_up(tail):
            return tail

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]
    if len(sentences) >= 2:
        tail = sentences[-1]
        if is_safe_follow_up(tail):
            return tail
    return None


def _follow_up_already_present(base: str, tail: str) -> bool:
    low_base = (base or "").lower()
    low_tail = (tail or "").strip().lower()
    if not low_tail:
        return True
    if low_tail in low_base:
        return True
    # Near-duplicate question stem.
    stem = re.sub(r"[^\w\s]", "", low_tail).strip()
    if stem and stem in re.sub(r"[^\w\s]", "", low_base):
        return True
    return False


def preserve_streamed_follow_up(replacement: str, streamed: str) -> str:
    """Append a safe streamed follow-up the user already saw after format retry."""
    base = (replacement or "").strip()
    provisional = strip_harmony_oss_artifacts(streamed or "").strip()
    if not base or not provisional:
        return base

    # Same-body extension (retry is a prefix of what streamed).
    if provisional.startswith(base) and len(provisional) > len(base):
        tail = provisional[len(base) :].strip()
        if tail and len(tail) <= 160:
            sentences = [part.strip() for part in _SENTENCE_SPLIT.split(tail) if part.strip()]
            if len(sentences) == 1 and is_safe_follow_up(sentences[0]):
                if not _follow_up_already_present(base, sentences[0]):
                    return f"{base}\n\n{sentences[0]}".strip()

    candidate = extract_follow_up_candidate(provisional)
    if not candidate or _follow_up_already_present(base, candidate):
        return base
    return f"{base}\n\n{candidate}".strip()


# Back-compat alias used by earlier fix.
merge_user_visible_stream_tail = preserve_streamed_follow_up
