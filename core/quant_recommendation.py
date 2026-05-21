"""Quantization recommendation heuristics for Model Manager GGUF picker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from core.gguf_quant import (
    ParsedQuant,
    is_iq_quant,
    parse_quant_from_gguf_path,
    quant_matches,
    quant_rank,
    rank_distance_to_preferred,
)
from core.model_params import infer_params_b, parse_params_b_from_label


class RecommendationConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QuantBadgeKind(str, Enum):
    NONE = "none"
    RECOMMENDED = "recommended"
    HIGHER_QUALITY = "higher_quality"
    LOWER_MEMORY = "lower_memory"
    QUALITY_FOCUSED = "quality_focused"
    BALANCED = "balanced"


_BADGE_TEXT: dict[QuantBadgeKind, str] = {
    QuantBadgeKind.NONE: "",
    QuantBadgeKind.RECOMMENDED: "Recommended",
    QuantBadgeKind.HIGHER_QUALITY: "Higher Quality",
    QuantBadgeKind.LOWER_MEMORY: "Lower Memory",
    QuantBadgeKind.QUALITY_FOCUSED: "Quality Focused",
    QuantBadgeKind.BALANCED: "Balanced",
}


@dataclass(frozen=True)
class SizeBandSpec:
    band_id: str
    primary: str
    secondary: str
    secondary_badge: QuantBadgeKind


_SIZE_BANDS: tuple[SizeBandSpec, ...] = (
    SizeBandSpec("tiny", "Q8_0", "Q6_K", QuantBadgeKind.LOWER_MEMORY),
    SizeBandSpec("small", "Q5_K_M", "Q6_K", QuantBadgeKind.HIGHER_QUALITY),
    SizeBandSpec("medium", "Q5_K_M", "Q6_K", QuantBadgeKind.HIGHER_QUALITY),
    SizeBandSpec("large_mid", "Q5_K_M", "Q6_K", QuantBadgeKind.HIGHER_QUALITY),
    SizeBandSpec("xlarge", "Q4_K_M", "Q5_K_M", QuantBadgeKind.HIGHER_QUALITY),
    SizeBandSpec("huge", "Q4_K_M", "Q5_K_M", QuantBadgeKind.HIGHER_QUALITY),
)

_UNKNOWN_BAND = SizeBandSpec("unknown", "Q5_K_M", "Q6_K", QuantBadgeKind.HIGHER_QUALITY)


@dataclass(frozen=True)
class BandResolution:
    band: SizeBandSpec
    params_b: float | None
    confidence: RecommendationConfidence
    confidence_score: float
    confidence_reason: str


@dataclass(frozen=True)
class ToolCallingModifier:
    score: float
    q5_bonus: float = 0.0
    q6_bonus: float = 0.0
    compress_penalty: float = 0.0
    apply_compress_penalty: bool = False


@dataclass(frozen=True)
class QuantRecommendationContext:
    repo_id: str
    title: str
    description: str
    params_b: float | None
    tool_calling_score: float
    hf_tags: tuple[str, ...]
    params_source: str = "unknown"


@dataclass(frozen=True)
class QuantFileRecommendation:
    path: str
    quant_label: str | None
    badge: QuantBadgeKind
    badge_text: str
    rationale: str
    is_default_pick: bool
    confidence: RecommendationConfidence
    confidence_score: float
    pick_score: float = 0.0


@dataclass(frozen=True)
class QuantRecommendationPlan:
    files: tuple[QuantFileRecommendation, ...]
    default_index: int | None
    summary_hint: str | None
    band_id: str
    plan_confidence: RecommendationConfidence
    plan_confidence_score: float
    confidence_reason: str
    primary_quant: str
    secondary_quant: str


class QuantHeuristicPlugin(Protocol):
    """Future VRAM / context-length plugins adjust per-file scores."""

    def adjust_score(
        self,
        context: QuantRecommendationContext,
        band: SizeBandSpec,
        path: str,
        parsed: ParsedQuant | None,
        base_score: float,
    ) -> float: ...


HEURISTIC_PLUGINS: list[QuantHeuristicPlugin] = []


def resolve_size_band(params_b: float | None) -> BandResolution:
    if params_b is None:
        return BandResolution(
            band=_UNKNOWN_BAND,
            params_b=None,
            confidence=RecommendationConfidence.LOW,
            confidence_score=0.35,
            confidence_reason="params_unknown",
        )
    pb = float(params_b)
    if pb <= 4:
        band = _SIZE_BANDS[0]
    elif pb <= 6:
        band = _SIZE_BANDS[1]
    elif pb <= 14:
        band = _SIZE_BANDS[2]
    elif pb < 30:
        band = _SIZE_BANDS[3]
    elif pb <= 70:
        band = _SIZE_BANDS[4]
    else:
        band = _SIZE_BANDS[5]
    return BandResolution(
        band=band,
        params_b=pb,
        confidence=RecommendationConfidence.HIGH,
        confidence_score=0.95,
        confidence_reason="params_resolved",
    )


def band_resolution_with_confidence(
    params_b: float | None,
    params_source: str,
) -> BandResolution:
    base = resolve_size_band(params_b)
    if params_source == "hf_card":
        conf = RecommendationConfidence.HIGH
        score = 0.92
        reason = "params_from_hf_card"
    elif params_source == "repo_inference":
        conf = RecommendationConfidence.MEDIUM
        score = 0.65
        reason = "params_inferred_repo_id"
    elif params_b is None:
        conf = RecommendationConfidence.LOW
        score = 0.35
        reason = "params_unknown"
    else:
        conf = base.confidence
        score = base.confidence_score
        reason = base.confidence_reason
    return BandResolution(
        band=base.band,
        params_b=base.params_b,
        confidence=conf,
        confidence_score=score,
        confidence_reason=reason,
    )


def apply_tool_calling_modifier(
    band: SizeBandSpec,
    tool_calling_score: float,
) -> ToolCallingModifier:
    s = max(0.0, min(1.0, float(tool_calling_score)))
    if s < 0.35:
        return ToolCallingModifier(score=s)
    if s >= 0.7:
        compress = band.band_id in ("tiny", "small", "medium", "large_mid")
        return ToolCallingModifier(
            score=s,
            q5_bonus=12.0,
            q6_bonus=6.0,
            compress_penalty=8.0,
            apply_compress_penalty=compress,
        )
    return ToolCallingModifier(score=s, q5_bonus=5.0, q6_bonus=2.0)


def _is_aggressive_compress(normalized: str) -> bool:
    u = normalized.upper()
    if u.startswith("IQ") and any(c in u for c in ("1", "2", "3")):
        return True
    if u.startswith("Q2") or u.startswith("Q3"):
        return True
    if u.startswith("Q4"):
        return True
    return False


def score_file_for_band(
    parsed: ParsedQuant | None,
    band: SizeBandSpec,
    modifier: ToolCallingModifier,
) -> float:
    if parsed is None:
        return 0.0
    n = parsed.normalized
    score = 0.0
    if quant_matches(band.primary, n):
        score = 100.0
    elif quant_matches(band.secondary, n):
        score = 70.0
    else:
        dist = rank_distance_to_preferred(parsed, band.primary)
        score = max(0.0, 40.0 - dist)

    if modifier.q5_bonus and quant_matches("Q5_K_M", n):
        score += modifier.q5_bonus
        if band.band_id in ("tiny", "small"):
            if modifier.score >= 0.7:
                score += 75.0
            elif modifier.score >= 0.35:
                score += 85.0
    if modifier.q6_bonus and quant_matches("Q6_K", n):
        score += modifier.q6_bonus
    if modifier.apply_compress_penalty and _is_aggressive_compress(n):
        score -= modifier.compress_penalty
    if is_iq_quant(parsed):
        score *= 0.85
    return score


def infer_tool_calling_score(
    *,
    capability_summary: dict[str, dict[str, Any]] | None = None,
    meta_capabilities: list[str] | None = None,
    hf_tags: list[str] | None = None,
    repo_id: str = "",
    title: str = "",
    description: str = "",
) -> float:
    score = 0.0
    cap = capability_summary or {}
    tool = cap.get("tool_use") or {}
    if bool(tool.get("value", False)):
        conf = float(tool.get("confidence", 0.6))
        score = max(score, min(1.0, conf))

    caps_text = " ".join(str(c).lower() for c in (meta_capabilities or []))
    if "tool use" in caps_text or "tool-use" in caps_text:
        score = max(score, 0.55)

    hay = " ".join(list(hf_tags or []) + [repo_id, title, description]).lower()
    needles = (
        "tool-use",
        "tool use",
        "function-calling",
        "function calling",
        "function call",
        "agent",
    )
    if any(n in hay for n in needles):
        score = max(score, 0.45)
    return min(1.0, score)


def build_context_from_hub_meta(
    *,
    repo_id: str,
    title: str,
    description: str,
    meta: dict[str, Any] | None = None,
    capability_summary: dict[str, dict[str, Any]] | None = None,
) -> QuantRecommendationContext:
    meta = meta or {}
    tags = tuple(str(t) for t in list(meta.get("hf_tags") or []) if str(t).strip())
    params_label = str(meta.get("params") or "")
    card = {k: meta.get(k) for k in ("params", "parameter_count", "parameters", "model_size")}
    params_b, source = infer_params_b(
        card=card,
        tags=list(tags),
        repo_id=repo_id,
        title=title,
        description=description,
        params_label=params_label if params_label and params_label != "Unknown" else None,
    )
    if params_b is None and params_label:
        params_b = parse_params_b_from_label(params_label)
        if params_b is not None:
            source = "hf_card"
    tool_score = infer_tool_calling_score(
        capability_summary=capability_summary,
        meta_capabilities=list(meta.get("capabilities") or []),
        hf_tags=list(tags),
        repo_id=repo_id,
        title=title,
        description=description,
    )
    return QuantRecommendationContext(
        repo_id=repo_id,
        title=title,
        description=description,
        params_b=params_b,
        tool_calling_score=tool_score,
        hf_tags=tags,
        params_source=source,
    )


def _badge_for_slot(
    normalized: str,
    band: SizeBandSpec,
) -> QuantBadgeKind:
    if quant_matches(band.primary, normalized):
        return QuantBadgeKind.RECOMMENDED
    if quant_matches(band.secondary, normalized):
        return band.secondary_badge
    return QuantBadgeKind.NONE


def _rationale_for_badge(
    badge: QuantBadgeKind,
    band: SizeBandSpec,
    tool_score: float,
    plan_confidence: RecommendationConfidence,
) -> str:
    prefix = ""
    if plan_confidence == RecommendationConfidence.LOW:
        prefix = "Likely good fit: "
    elif plan_confidence == RecommendationConfidence.MEDIUM:
        prefix = ""

    tool_clause = ""
    if tool_score >= 0.35 and badge == QuantBadgeKind.RECOMMENDED:
        if tool_score >= 0.7:
            tool_clause = " Also well-suited for tool calling and structured outputs."
        else:
            tool_clause = " Reasonable for tool use when available in this repo."

    if badge == QuantBadgeKind.RECOMMENDED:
        by_band = {
            "tiny": "Maximum quality for very small models; avoids overly aggressive compression.",
            "small": "Best balance of quality and memory for compact models.",
            "medium": "Best balance of quality and VRAM for models this size.",
            "large_mid": "Balanced choice for mid-size models (15B–29B class).",
            "xlarge": "Prioritizes VRAM efficiency for large models while keeping solid output quality.",
            "huge": "VRAM-friendly default for very large models.",
            "unknown": "Balanced default when model size could not be confirmed.",
        }
        return prefix + by_band.get(band.band_id, by_band["medium"]) + tool_clause
    if badge == QuantBadgeKind.HIGHER_QUALITY:
        return prefix + "More headroom for accuracy; uses more memory than the primary pick."
    if badge == QuantBadgeKind.LOWER_MEMORY:
        return prefix + "Lower memory usage than the primary pick; slightly lower fidelity."
    return ""


def _alternative_rationale() -> str:
    return "Alternative quantization — compare file size and your hardware budget before downloading."


def recommend_quants(
    context: QuantRecommendationContext,
    files: list[tuple[str, int | None]],
) -> QuantRecommendationPlan:
    resolution = band_resolution_with_confidence(context.params_b, context.params_source)
    band = resolution.band
    modifier = apply_tool_calling_modifier(band, context.tool_calling_score)

    plan_conf = resolution.confidence
    plan_score = resolution.confidence_score
    if context.tool_calling_score >= 0.35 and context.tool_calling_score < 0.7:
        if plan_conf == RecommendationConfidence.HIGH:
            plan_conf = RecommendationConfidence.MEDIUM
            plan_score = min(plan_score, 0.78)

    scored: list[tuple[int, str, int | None, ParsedQuant | None, float]] = []
    for idx, (path, size_b) in enumerate(files):
        parsed = parse_quant_from_gguf_path(path)
        sc = score_file_for_band(parsed, band, modifier)
        for plugin in HEURISTIC_PLUGINS:
            sc = plugin.adjust_score(context, band, path, parsed, sc)
        scored.append((idx, path, size_b, parsed, sc))

    best_idx: int | None = None
    best_score = -1.0

    def _prefer_over_current(
        new_score: float,
        new_parsed: ParsedQuant | None,
        cur_parsed: ParsedQuant | None,
    ) -> bool:
        if new_score > best_score:
            return True
        if new_score < best_score:
            return False
        if (
            modifier.score >= 0.35
            and band.band_id in ("tiny", "small")
            and new_parsed is not None
            and quant_matches("Q5_K_M", new_parsed.normalized)
        ):
            return cur_parsed is None or not quant_matches("Q5_K_M", cur_parsed.normalized)
        return False

    cur_parsed: ParsedQuant | None = None
    for idx, _path, _sz, parsed, sc in scored:
        if _prefer_over_current(sc, parsed, cur_parsed):
            best_score = sc
            best_idx = idx
            cur_parsed = parsed

    recs: list[QuantFileRecommendation] = []
    for idx, path, _sz, parsed, sc in scored:
        label = parsed.normalized if parsed else None
        badge = QuantBadgeKind.NONE
        if label:
            badge = _badge_for_slot(label, band)
        badge_text = _BADGE_TEXT.get(badge, "")
        is_default = best_idx is not None and idx == best_idx and sc > 0
        file_conf = plan_conf
        file_score = plan_score
        if badge == QuantBadgeKind.NONE and is_default:
            file_conf = RecommendationConfidence.LOW
            file_score = 0.3
        elif badge == QuantBadgeKind.NONE:
            file_conf = RecommendationConfidence.LOW
            file_score = 0.25
        elif is_default and rank_distance_to_preferred(parsed, band.primary) > 0:
            file_conf = RecommendationConfidence.LOW
            file_score = min(file_score, 0.4)

        if badge != QuantBadgeKind.NONE:
            rationale = _rationale_for_badge(badge, band, context.tool_calling_score, plan_conf)
        elif is_default:
            rationale = _alternative_rationale()
        else:
            rationale = ""

        recs.append(
            QuantFileRecommendation(
                path=path,
                quant_label=label,
                badge=badge,
                badge_text=badge_text,
                rationale=rationale,
                is_default_pick=is_default,
                confidence=file_conf,
                confidence_score=file_score,
                pick_score=sc,
            )
        )

    default_combo_index: int | None = None
    if best_idx is not None:
        default_combo_index = best_idx + 1

    primary = band.primary
    secondary = band.secondary
    hint = (
        f"{len(files)} file(s) available. "
        f"We suggest {primary} first"
        + (f" or {secondary}." if secondary else ".")
        + " Choose a quantization, then Download."
    )
    return QuantRecommendationPlan(
        files=tuple(recs),
        default_index=default_combo_index,
        summary_hint=hint,
        band_id=band.band_id,
        plan_confidence=plan_conf,
        plan_confidence_score=plan_score,
        confidence_reason=resolution.confidence_reason,
        primary_quant=primary,
        secondary_quant=secondary,
    )
