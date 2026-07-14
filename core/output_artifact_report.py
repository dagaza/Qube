"""Detect and log template scaffold artifacts in model completion text."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

from core.native_llm_debug import llm_debug_enabled

if TYPE_CHECKING:
    from core.template_output_profile import TemplateOutputProfile

logger = logging.getLogger("Qube.NativeLLM.Debug")

ArtifactType = Literal[
    "assistant_open",
    "assistant_close",
    "thinking_open",
    "thinking_close",
    "harmony_control",
    "chatml_marker",
    "mistral_marker",
    "other_control",
]

ParsePath = Literal["harmony_channel", "delimiter_grammar", "fallback_strip", "none"]
ParseConfidence = Literal["high", "low"]

_HARMONY_CONTROL_RE = re.compile(
    r"<\|start\|>|<\|end\|>|<\|return\|>|<\|channel\|>|<\|message\|>|<\|final\|>",
    re.I,
)
_CHATML_MARKER_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I)
_MISTRAL_MARKER_RE = re.compile(r"\[/?INST\]|</s>", re.I)
_ASSISTANT_OPEN_RE = re.compile(r"<\|assistant\|>", re.I)
_ASSISTANT_CLOSE_RE = re.compile(r"</\|assistant\|>", re.I)
_THINKING_TAG_RE = re.compile(
    r"(?is)<(?:redacted_)?think(?:ing)?>|</(?:redacted_)?think(?:ing)?>"
)


@dataclass(frozen=True)
class DetectedArtifact:
    artifact_type: ArtifactType
    raw: str
    start: int
    end: int


@dataclass
class OutputArtifactReport:
    template_family: str
    grammar_tier: str
    artifact_detected: bool
    artifacts: list[DetectedArtifact] = field(default_factory=list)
    parse_path: ParsePath = "none"
    parse_confidence: ParseConfidence = "low"
    raw_len: int = 0
    visible_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "output_artifact_report",
            "template_family": self.template_family,
            "grammar_tier": self.grammar_tier,
            "artifact_detected": self.artifact_detected,
            "artifact_count": len(self.artifacts),
            "artifacts": [
                {
                    "type": a.artifact_type,
                    "raw": a.raw,
                    "start": a.start,
                    "end": a.end,
                }
                for a in self.artifacts
            ],
            "parse_path": self.parse_path,
            "parse_confidence": self.parse_confidence,
            "raw_len": self.raw_len,
            "visible_len": self.visible_len,
        }


def _classify_token(raw: str) -> ArtifactType:
    if _ASSISTANT_OPEN_RE.fullmatch(raw):
        return "assistant_open"
    if _ASSISTANT_CLOSE_RE.fullmatch(raw):
        return "assistant_close"
    if re.fullmatch(r"(?is)<(?:redacted_)?think(?:ing)?>", raw):
        return "thinking_open"
    if re.fullmatch(r"(?is)</(?:redacted_)?think(?:ing)?>", raw):
        return "thinking_close"
    if _HARMONY_CONTROL_RE.search(raw):
        return "harmony_control"
    if _CHATML_MARKER_RE.search(raw):
        return "chatml_marker"
    if _MISTRAL_MARKER_RE.search(raw):
        return "mistral_marker"
    return "other_control"


def _patterns_for_profile(profile: Optional["TemplateOutputProfile"]) -> list[tuple[re.Pattern[str], ArtifactType]]:
    patterns: list[tuple[re.Pattern[str], ArtifactType]] = []
    if profile is not None:
        for tok in profile.assistant_open_tokens:
            patterns.append((re.compile(re.escape(tok), re.I), "assistant_open"))
        for tok in profile.assistant_close_tokens:
            patterns.append((re.compile(re.escape(tok), re.I), "assistant_close"))
        for tok in profile.thinking_open_tokens:
            patterns.append((re.compile(re.escape(tok), re.I), "thinking_open"))
        for tok in profile.thinking_close_tokens:
            patterns.append((re.compile(re.escape(tok), re.I), "thinking_close"))
    patterns.extend(
        [
            (_ASSISTANT_OPEN_RE, "assistant_open"),
            (_ASSISTANT_CLOSE_RE, "assistant_close"),
            (_THINKING_TAG_RE, "thinking_open"),
            (_HARMONY_CONTROL_RE, "harmony_control"),
            (_CHATML_MARKER_RE, "chatml_marker"),
            (_MISTRAL_MARKER_RE, "mistral_marker"),
        ]
    )
    return patterns


def detect_output_artifacts(
    text: str,
    *,
    profile: Optional["TemplateOutputProfile"] = None,
) -> list[DetectedArtifact]:
    """Return structured artifact hits found in ``text``."""
    if not text:
        return []
    found: list[DetectedArtifact] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern, default_type in _patterns_for_profile(profile):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            raw = match.group(0)
            artifact_type = default_type if default_type != "thinking_open" else _classify_token(raw)
            found.append(
                DetectedArtifact(
                    artifact_type=artifact_type,
                    raw=raw,
                    start=span[0],
                    end=span[1],
                )
            )
    found.sort(key=lambda a: a.start)
    return found


def build_output_artifact_report(
    *,
    raw_text: str,
    visible_text: str,
    profile: Optional["TemplateOutputProfile"] = None,
    parse_path: ParsePath = "none",
    parse_confidence: ParseConfidence = "low",
) -> OutputArtifactReport:
    artifacts = detect_output_artifacts(raw_text, profile=profile)
    family = profile.family if profile is not None else "unknown"
    tier = profile.grammar_tier if profile is not None else "none"
    confidence: ParseConfidence = parse_confidence
    if artifacts and parse_path in ("delimiter_grammar", "harmony_channel") and not (visible_text or "").strip():
        confidence = "low"
    elif artifacts and (raw_text or "").strip() == (visible_text or "").strip():
        confidence = "low"
    elif parse_path in ("delimiter_grammar", "harmony_channel") and not artifacts:
        confidence = "high"
    return OutputArtifactReport(
        template_family=family,
        grammar_tier=tier,
        artifact_detected=bool(artifacts),
        artifacts=artifacts,
        parse_path=parse_path,
        parse_confidence=confidence,
        raw_len=len(raw_text or ""),
        visible_len=len(visible_text or ""),
    )


def log_output_artifact_report(report: OutputArtifactReport) -> None:
    """Emit structured artifact telemetry when debug is on or artifacts were found."""
    if not report.artifact_detected and not llm_debug_enabled():
        return
    payload = report.to_dict()
    logger.info(json.dumps(payload, ensure_ascii=False))
    if report.artifact_detected:
        types = [a.artifact_type for a in report.artifacts]
        logger.info(
            "[OutputArtifactReport] family=%s tier=%s count=%d types=%s path=%s",
            report.template_family,
            report.grammar_tier,
            len(report.artifacts),
            types[:8],
            report.parse_path,
        )


def artifact_types_for_validation(artifacts: Sequence[DetectedArtifact]) -> list[str]:
    """Map detected artifacts to validation issue hints."""
    issues: list[str] = []
    for art in artifacts:
        if art.artifact_type in ("assistant_open", "assistant_close"):
            issues.append("template_leakage")
        elif art.artifact_type in ("thinking_open", "thinking_close"):
            issues.append("template_leakage")
        elif art.artifact_type == "harmony_control":
            issues.append("template_leakage")
        elif art.artifact_type in ("chatml_marker", "mistral_marker"):
            issues.append("template_leakage")
    return list(dict.fromkeys(issues))


__all__ = [
    "ArtifactType",
    "DetectedArtifact",
    "OutputArtifactReport",
    "ParseConfidence",
    "ParsePath",
    "artifact_types_for_validation",
    "build_output_artifact_report",
    "detect_output_artifacts",
    "log_output_artifact_report",
]
