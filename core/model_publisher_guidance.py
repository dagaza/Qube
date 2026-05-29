"""
Deterministic extraction of publisher README contract signals (reasoning tags, defaults).

Does not parse or apply full README system-prompt presets — contract metadata only.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from core.model_capability_detection import match_pattern, normalize_model_id

ReasoningDefault = Literal["off", "on", "unknown"]

SOURCE_WEIGHTS: dict[str, float] = {
    "curated": 4.0,
    "curated_pattern": 3.5,
    "readme": 2.5,
}

THINKING_TAG_PATTERNS: tuple[str, ...] = (
    "<think>",
    "</think>",
    "<thinking>",
    "</thinking>",
)

CHAT_TEMPLATE_MAP: dict[str, str] = {
    "llama 3 chat": "llama3",
    "llama-3 chat": "llama3",
    "llama3 chat": "llama3",
    "command-r": "mistral",
    "command r": "mistral",
    "chatml": "chatml",
    "mistral instruct": "mistral",
}

_SECTION_HEADER_RE = re.compile(
    r"^#{1,6}\s*(.+)$|^(system\s+role|system\s+prompt|reasoning|how\s+to\s+set)\b",
    re.IGNORECASE | re.MULTILINE,
)

_PRESET_SECTION_MARKERS: tuple[str, ...] = (
    "multi-tiered",
    "creative simple",
    "creative advanced",
    "creative multi-tiered",
    "system prompts available",
)


def _count_ignored_preset_sections(text: str, min_chars: int = 400) -> int:
    """Count long publisher preset sections (often plain text, not fenced)."""
    low = (text or "").lower()
    count = 0
    for marker in _PRESET_SECTION_MARKERS:
        idx = low.find(marker)
        if idx < 0:
            continue
        chunk = low[idx : idx + min_chars + 200]
        if len(chunk.strip()) >= min_chars:
            count += 1
    return count


@dataclass(frozen=True)
class PublisherGuidance:
    thinking_tags: tuple[str, ...]
    default_reasoning_without_system: ReasoningDefault
    reasoning_controlled_by_system: bool
    mentioned_chat_templates: tuple[str, ...]
    confidence: float
    source: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> PublisherGuidance:
        return PublisherGuidance(
            thinking_tags=tuple(str(x) for x in (raw.get("thinking_tags") or [])),
            default_reasoning_without_system=_coerce_reasoning_default(
                raw.get("default_reasoning_without_system")
            ),
            reasoning_controlled_by_system=bool(raw.get("reasoning_controlled_by_system", False)),
            mentioned_chat_templates=tuple(
                str(x) for x in (raw.get("mentioned_chat_templates") or [])
            ),
            confidence=float(raw.get("confidence", 0.0)),
            source=str(raw.get("source") or "unknown"),
            evidence=tuple(str(x) for x in (raw.get("evidence") or [])),
        )


def _coerce_reasoning_default(value: Any) -> ReasoningDefault:
    s = str(value or "").strip().lower()
    if s in ("off", "on"):
        return s  # type: ignore[return-value]
    return "unknown"


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _count_large_fenced_blocks(text: str, min_chars: int = 400) -> int:
    count = 0
    for m in _FENCE_RE.finditer(text or ""):
        if len(m.group(0)) >= min_chars:
            count += 1
    return count


def _detect_thinking_tags(text: str) -> list[str]:
    low = (text or "").lower()
    found: list[str] = []
    for tag in THINKING_TAG_PATTERNS:
        if tag.lower() in low and tag not in found:
            found.append(tag)
    return found


def _detect_default_reasoning_off(text: str) -> bool:
    low = (text or "").lower()
    no_system = (
        "do not set" in low
        and "system prompt" in low
    ) or (
        "without" in low
        and "system prompt" in low
        and "off" in low
    )
    reasoning_off = (
        "reasoning" in low or "thinking" in low
    ) and (
        "off by default" in low
        or "will be off" in low
        or "reasoning/thinking will be off" in low
        or ("off" in low and "default" in low and ("reasoning" in low or "thinking" in low))
    )
    return bool(no_system and reasoning_off) or bool(
        "if you do not set" in low
        and "system prompt" in low
        and ("off" in low or "off by default" in low)
        and ("reasoning" in low or "thinking" in low)
    )


def _detect_reasoning_controlled_by_system(text: str) -> bool:
    low = (text or "").lower()
    has_system = "system prompt" in low or "system role" in low or "system message" in low
    has_control = "control" in low or "root access" in low
    has_reasoning = "reasoning" in low or "thinking" in low
    return bool(has_system and has_control and has_reasoning)


def _detect_mentioned_templates(text: str) -> list[str]:
    low = (text or "").lower()
    found: list[str] = []
    for phrase, family in CHAT_TEMPLATE_MAP.items():
        if phrase in low and family not in found:
            found.append(family)
    return found


def _has_relevant_section(text: str) -> bool:
    if not text:
        return False
    for m in _SECTION_HEADER_RE.finditer(text):
        if m.group(0):
            return True
    low = text.lower()
    return any(
        k in low
        for k in (
            "system prompt",
            "system role",
            "reasoning on",
            "reasoning off",
            "how to set",
        )
    )


def extract_publisher_guidance(readme: str) -> PublisherGuidance | None:
    """
    Extract contract signals from README markdown. Returns None when no signals found.
    """
    raw = str(readme or "").strip()
    if not raw:
        return None

    evidence: list[str] = []
    thinking_tags = _detect_thinking_tags(raw)
    if thinking_tags:
        evidence.append("thinking_tags")

    default_off = _detect_default_reasoning_off(raw)
    if default_off:
        evidence.append("default_reasoning_off")

    system_controlled = _detect_reasoning_controlled_by_system(raw)
    if system_controlled:
        evidence.append("reasoning_system_controlled")

    templates = _detect_mentioned_templates(raw)
    if templates:
        evidence.append("mentioned_templates")

    large_blocks = _count_large_fenced_blocks(raw) + _count_ignored_preset_sections(raw)
    if large_blocks:
        evidence.append(f"ignored_preset_block:{large_blocks}")

    if not evidence or (
        evidence == [f"ignored_preset_block:{large_blocks}"] and not thinking_tags
    ):
        if not _has_relevant_section(raw) and not thinking_tags:
            return None

    if not thinking_tags and not default_off and not system_controlled and not templates:
        return None

    confidence = 0.5
    if thinking_tags:
        confidence += 0.15
    if default_off:
        confidence += 0.15
    if system_controlled:
        confidence += 0.1
    if templates:
        confidence += 0.05
    confidence = min(0.95, confidence)

    default_reasoning: ReasoningDefault = "unknown"
    if default_off:
        default_reasoning = "off"

    return PublisherGuidance(
        thinking_tags=tuple(thinking_tags),
        default_reasoning_without_system=default_reasoning,
        reasoning_controlled_by_system=system_controlled,
        mentioned_chat_templates=tuple(templates),
        confidence=confidence,
        source="readme",
        evidence=tuple(evidence),
    )


def _guidance_from_curated_dict(raw: dict[str, Any], source: str) -> PublisherGuidance:
    tags = tuple(str(x) for x in (raw.get("thinking_tags") or []))
    templates = tuple(str(x) for x in (raw.get("mentioned_chat_templates") or []))
    return PublisherGuidance(
        thinking_tags=tags,
        default_reasoning_without_system=_coerce_reasoning_default(
            raw.get("default_reasoning_without_system")
        ),
        reasoning_controlled_by_system=bool(raw.get("reasoning_controlled_by_system", False)),
        mentioned_chat_templates=templates,
        confidence=float(raw.get("confidence", 0.9 if source == "curated" else 0.85)),
        source=source,
        evidence=(f"curated:{source}",),
    )


def lookup_curated_publisher_guidance(
    registry: dict[str, Any],
    *,
    model_id: str,
    normalized_model_id: str,
    model_name: str = "",
) -> PublisherGuidance | None:
    """Resolve curated exact/pattern publisher guidance from registry."""
    pg = registry.get("publisher_guidance") or {}
    if not isinstance(pg, dict):
        return None

    exact = pg.get("exact") or {}
    if isinstance(exact, dict):
        mid = str(model_id or "").strip().lower()
        tail = mid.split("/")[-1] if mid else ""
        for key in (mid, tail, normalized_model_id):
            if key and key in exact and isinstance(exact[key], dict):
                return _guidance_from_curated_dict(exact[key], "curated")

    patterns = list(pg.get("patterns") or [])
    name = str(model_name or "").lower()
    raw_id = str(model_id or "").lower()
    for p in patterns:
        if not isinstance(p, dict):
            continue
        guidance = p.get("guidance") or {}
        if not isinstance(guidance, dict):
            continue
        did_match = (
            match_pattern(raw_id, name, p)
            or match_pattern(normalized_model_id, name, p)
            or match_pattern(raw_id.split("/")[-1], name, p)
        )
        if did_match:
            return _guidance_from_curated_dict(guidance, "curated_pattern")
    return None


def merge_publisher_guidance(
    *sources: PublisherGuidance | None,
) -> PublisherGuidance | None:
    """Pick highest-weight source; merge tags/templates from all present sources."""
    present = [s for s in sources if s is not None]
    if not present:
        return None

    winner = max(
        present,
        key=lambda s: (SOURCE_WEIGHTS.get(s.source, 1.0), s.confidence),
    )

    all_tags: list[str] = []
    all_templates: list[str] = []
    all_evidence: list[str] = []
    for s in present:
        for t in s.thinking_tags:
            if t not in all_tags:
                all_tags.append(t)
        for t in s.mentioned_chat_templates:
            if t not in all_templates:
                all_templates.append(t)
        for e in s.evidence:
            if e not in all_evidence:
                all_evidence.append(e)

    default = winner.default_reasoning_without_system
    for s in sorted(present, key=lambda x: SOURCE_WEIGHTS.get(x.source, 1.0), reverse=True):
        if s.default_reasoning_without_system != "unknown":
            default = s.default_reasoning_without_system
            break

    system_controlled = any(s.reasoning_controlled_by_system for s in present)

    return PublisherGuidance(
        thinking_tags=tuple(all_tags),
        default_reasoning_without_system=default,
        reasoning_controlled_by_system=system_controlled,
        mentioned_chat_templates=tuple(all_templates),
        confidence=winner.confidence,
        source=winner.source,
        evidence=tuple(all_evidence),
    )


def apply_guidance_to_reasoning_profile(
    profile: Any,
    guidance: PublisherGuidance | None,
) -> Any:
    """
    Boost reasoning profile metadata when README/curated guidance confirms thinking tags
    but GGUF tokenizer scan did not find vocab hits. Returns same profile type (mutated copy).
    """
    if guidance is None or not guidance.thinking_tags:
        return profile
    if profile is None:
        return profile

    method = str(getattr(profile, "detection_method", "") or "")
    if method == "tokenizer_scan":
        return profile

    from dataclasses import replace

    patterns = list(getattr(profile, "thinking_token_patterns", None) or [])
    for tag in guidance.thinking_tags:
        if tag not in patterns:
            patterns.append(tag)

    new_conf = min(0.75, max(float(getattr(profile, "reasoning_confidence", 0.0)), guidance.confidence))
    suffix = "readme_guidance" if method else "readme_guidance"
    new_method = f"{method}+{suffix}" if method and "readme_guidance" not in method else suffix

    return replace(
        profile,
        supports_thinking_tokens=True,
        thinking_token_patterns=patterns,
        reasoning_confidence=new_conf,
        detection_method=new_method,
    )
