"""Remove Harmony / OSS-style template scaffolding from model text (runtime output)."""
from __future__ import annotations

import re

from core.gemma_output_strip import strip_gemma_output_artifacts
from core.harmony_degeneration import polish_harmony_visible_text

# Log-derived bridge: <|end|><|start|>assistant<|channel|>final<|message|>
_HARMONY_BRIDGE = re.compile(
    r"(?is)<\|end\|>\s*<\|start\|>\s*assistant\s*<\|channel\|>\s*final\s*<\|message\|>",
)
# Instruction-echo preface often preceding the bridge (one shot, non-greedy).
_INSTRUCTION_ECHO = re.compile(
    r"(?is)We\s+need\s+to\s+explain.*?Provide\s+concise\.?\s*",
)
# Untagged Harmony planning preface at the start of a leaked final-channel reply.
_PLANNING_PREFACE = re.compile(
    r"(?is)^\s*(?:let'?s\s+clarify(?:\s+that)?\.?\s*)?"
    r"(?:the\s+user\s+says\b[^.!?\n]{0,400}[.!?:]?\s*)+"
)
# Untagged OSS/Harmony scratchpad tail observed after a partial final answer.
_SCRATCHPAD_TAIL = re.compile(
    r"(?is)\s*(?:[?.!…]{3,}\s*)?(?:"
    r"We\s+need\s+to\s+(?:answer|explain)|"
    r"We\s+have\s+to\s+answer|"
    r"We\s+should\s+produce|"
    r"We\s+have\s+sources?|"
    r"Source\s+\d+\s+(?:indicates|says)|"
    r"Let's\s+(?:clarify|produce\s+answer)|"
    r"The\s+user\s+says\b|"
    r"The\s+question\s+says\b|"
    r"The\s+user\s+wants\b|"
    r"They\s+may\s+be\s+asking\b|"
    r"no\s+meta\s+commentary\b"
    r").*$",
)
# Keep the natural answer if a stop/scratchpad cut leaves punctuation noise at the end.
_TRAILING_NOISE = re.compile(r"(?s)(?:\s*[?.!…]){3,}\s*$")
# Planning-only residue after preface/tail stripping (dots, dashes, whitespace).
_PUNCT_ONLY_REMAINDER = re.compile(r"^[\s.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]+$")
# Spaced punctuation loops from gpt-oss can contain NBSP/narrow NBSP between ellipses.
_DEGENERATE_PUNCT_SEGMENT = re.compile(
    r"[\s\u00a0\u202f\u2009\u200a\u2028.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{18,}"
)
_PUNCT_CHARS = set(".…?\u2026\u2010\u2011\u2012\u2013\u2014-")
_SENTENCE_END = re.compile(r"[.!?。！？](?:[\"'”’)\]]+)?")
_META_COMMAND_PREFIX = re.compile(
    r"(?is)^\s*provide\s+final\s+answer\b\s*(?:[.!?:\-–—]+|\n+|\s+)?"
)
# Any remaining <|...|> control tokens (bounded label).
_CONTROL_TOKEN = re.compile(r"<\|[^|\n]{1,56}\|>")
# Partial / malformed Harmony tokens (ChatML-on-Jinja drift, broken delimiters).
_MALFORMED_CONTROL = re.compile(
    r"(?i)<\|?channel\|?>|<\|?message\|?>|<\|?start\|?>|<\|?final\|?>|<\|?end\|?>"
)
# Channel scaffold tail after an otherwise complete answer.
_CHANNEL_TAIL = re.compile(r"(?is)\n+\s*<\|?channel\|?>.*$")
# Mistral instruct markers leaked when prompt anchor was wrong or the model continues the template.
_MISTRAL_INST_MARKERS = re.compile(r"\s*\[/?INST\]\s*")
_MISTRAL_EOS_TAIL = re.compile(r"\s*</s>\s*$")


def _strip_degenerate_punctuation_tail(text: str) -> str:
    """Drop gpt-oss punctuation filler and any incomplete clause before it."""
    if not text:
        return text
    for match in _DEGENERATE_PUNCT_SEGMENT.finditer(text):
        segment = match.group(0)
        punct_count = sum(1 for ch in segment if ch in _PUNCT_CHARS)
        if punct_count < 8:
            continue
        cut = text[: match.start()].rstrip()
        last_sentence_end = None
        for sentence_match in _SENTENCE_END.finditer(cut):
            last_sentence_end = sentence_match.end()
        if last_sentence_end is not None and last_sentence_end < len(cut):
            return cut[:last_sentence_end].rstrip()
        return cut
    return text


def strip_mistral_instruct_artifacts(text: str) -> str:
    """Remove Mistral template markers without altering inter-token spacing.

    Must preserve leading/trailing spaces on each fragment — streaming passes one
    delta at a time to TTS; a global ``strip()`` merges words (``Yes`` + `` world`` → ``Yesworld``).
    """
    if not text:
        return text
    if "[INST]" not in text and "[/INST]" not in text and "</s>" not in text:
        return text
    t = _MISTRAL_INST_MARKERS.sub(" ", text)
    return _MISTRAL_EOS_TAIL.sub("", t)


def strip_harmony_oss_artifacts(text: str) -> str:
    if not text or not text.strip():
        return text
    t = polish_harmony_visible_text(text)
    t = _INSTRUCTION_ECHO.sub("", text, count=1)
    t = _PLANNING_PREFACE.sub("", t, count=1)
    t = _HARMONY_BRIDGE.sub("", t)
    t = _CHANNEL_TAIL.sub("", t)
    t = _CONTROL_TOKEN.sub("", t)
    t = _MALFORMED_CONTROL.sub("", t)
    t = _META_COMMAND_PREFIX.sub("", t, count=1)
    t = _SCRATCHPAD_TAIL.sub("", t, count=1)
    t = _strip_degenerate_punctuation_tail(t)
    t = _TRAILING_NOISE.sub("", t)
    t = strip_mistral_instruct_artifacts(t)
    if t and _PUNCT_ONLY_REMAINDER.match(t.strip()):
        return ""
    return strip_gemma_output_artifacts(t)


def merge_user_visible_stream_tail(replacement: str, streamed: str) -> str:
    """Keep a short streamed follow-up the user already saw after format retry."""
    from core.conversational_follow_up import preserve_streamed_follow_up

    return preserve_streamed_follow_up(replacement, streamed)
