"""Delimiter-grammar output extraction for Nemotron/Phi-style scaffold tokens."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.output_artifact_report import DetectedArtifact, detect_output_artifacts
from core.redacted_thinking_filter import _longest_suffix_that_is_prefix_of

if TYPE_CHECKING:
    from core.template_output_profile import TemplateOutputProfile

_STRIP_RE_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _strip_pattern(tokens: tuple[str, ...]) -> re.Pattern[str]:
    key = tokens
    cached = _STRIP_RE_CACHE.get(key)
    if cached is not None:
        return cached
    parts = [re.escape(tok) for tok in tokens if tok]
    if not parts:
        pat = re.compile(r"a^")
    else:
        pat = re.compile("|".join(parts), re.I)
    _STRIP_RE_CACHE[key] = pat
    return pat


@dataclass
class ParsedCompletion:
    visible_text: str
    discarded_artifacts: list[DetectedArtifact] = field(default_factory=list)
    parse_path: str = "delimiter_grammar"
    parse_confidence: str = "high"


def _longest_suffix_scaffold_prefix(text: str, needles: tuple[str, ...]) -> int:
    best = 0
    for needle in needles:
        best = max(best, _longest_suffix_that_is_prefix_of(text, needle))
    if best:
        return best
    return _longest_suffix_that_is_prefix_of(text, "<|")


def extract_delimiter_grammar(
    text: str,
    profile: "TemplateOutputProfile",
) -> ParsedCompletion:
    """Final-pass extraction: remove scaffold tokens and return visible payload."""
    raw = text or ""
    if not raw.strip():
        return ParsedCompletion(visible_text=raw, parse_confidence="high")
    artifacts = detect_output_artifacts(raw, profile=profile)
    scaffold = profile.scaffold_tokens()
    if not scaffold:
        return ParsedCompletion(
            visible_text=raw,
            discarded_artifacts=artifacts,
            parse_confidence="low",
        )
    visible = _strip_pattern(scaffold).sub("", raw)
    visible = re.sub(r"\n{3,}", "\n\n", visible).strip()
    confidence = "high" if artifacts else "low"
    return ParsedCompletion(
        visible_text=visible,
        discarded_artifacts=artifacts,
        parse_confidence=confidence,
    )


class DelimiterGrammarStreamFilter:
    """Chunk-safe delimiter grammar extractor for live streaming."""

    __slots__ = ("_profile", "_needles", "_hold", "_strip_re")

    def __init__(self, profile: "TemplateOutputProfile") -> None:
        self._profile = profile
        self._needles = profile.scaffold_tokens()
        self._hold = ""
        self._strip_re = _strip_pattern(self._needles)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        combined = self._hold + chunk
        self._hold = ""
        cleaned = self._strip_re.sub("", combined)
        hold = _longest_suffix_scaffold_prefix(cleaned, self._needles)
        if hold:
            self._hold = cleaned[-hold:]
            return cleaned[:-hold]
        return cleaned

    def flush(self) -> str:
        if not self._hold:
            return ""
        tail = self._strip_re.sub("", self._hold)
        self._hold = ""
        return tail


__all__ = [
    "DelimiterGrammarStreamFilter",
    "ParsedCompletion",
    "extract_delimiter_grammar",
]
