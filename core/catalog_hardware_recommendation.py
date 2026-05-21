"""Rank Qube Verified catalog entries against the local hardware capability profile."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from core.hardware_capability_profile import (
    HardwareCapabilityProfile,
    HardwareTier,
    detect_hardware_capability_profile,
)
from core.model_params import infer_params_b
from core.qube_verified_models import CatalogEntry, load_qube_verified_models

# Rough Q4_K_M footprint (GB) per billion parameters — conservative for fit checks.
_GB_PER_BILLION_Q4 = 0.62

_MOE_ACTIVE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*b\s+a(\d+(?:\.\d+)?)\s*b",
    re.IGNORECASE,
)

_PHI_MODEL_RE = re.compile(r"phi[-\s]?(\d+(?:\.\d+)?)", re.IGNORECASE)

_LAPTOP_HINTS = (
    "laptop",
    "lightweight",
    "ultra-efficient",
    "low-vram",
    "compact",
    "mini",
)


class CatalogFitLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CatalogFitAssessment:
    catalog_id: str
    title: str
    fit_level: CatalogFitLevel
    score: float
    params_b: float | None
    estimated_q4_gb: float | None
    rationale: str


@dataclass(frozen=True)
class CatalogRecommendationPlan:
    profile: HardwareCapabilityProfile
    assessments: tuple[CatalogFitAssessment, ...]
    recommended: tuple[CatalogFitAssessment, ...]
    banner_text: str
    detail_text: str


def infer_effective_params_b(
    *,
    title: str,
    description: str = "",
    repo_id: str = "",
) -> float | None:
    hay = " ".join([title, description, repo_id])
    m = _MOE_ACTIVE_RE.search(hay)
    if m:
        return float(m.group(2))
    phi = _PHI_MODEL_RE.search(hay)
    if phi:
        return float(phi.group(1))
    params_b, _source = infer_params_b(
        card={},
        tags=[],
        repo_id=repo_id,
        title=title,
        description=description,
    )
    return params_b


def estimate_q4_footprint_gb(params_b: float | None) -> float | None:
    if params_b is None or params_b <= 0:
        return None
    return float(params_b) * _GB_PER_BILLION_Q4


def _fit_level_for_ratio(ratio: float, params_known: bool) -> CatalogFitLevel:
    if not params_known:
        return CatalogFitLevel.UNKNOWN
    if ratio <= 0.85:
        return CatalogFitLevel.EXCELLENT
    if ratio <= 1.0:
        return CatalogFitLevel.GOOD
    if ratio <= 1.35:
        return CatalogFitLevel.MARGINAL
    return CatalogFitLevel.POOR


def _tier_bonus(entry: CatalogEntry, profile: HardwareCapabilityProfile) -> float:
    hay = f"{entry.title} {entry.description} {entry.catalog_id}".lower()
    bonus = 0.0
    if profile.tier in (HardwareTier.COMPACT, HardwareTier.STANDARD):
        if any(h in hay for h in _LAPTOP_HINTS):
            bonus += 8.0
        if entry.catalog_id in ("gemma-4-e4b-it", "phi-4-mini-instruct"):
            bonus += 6.0
    if profile.tier == HardwareTier.ENTHUSIAST and "high-end" in hay:
        bonus += 4.0
    return bonus


def _score_entry(
    entry: CatalogEntry,
    profile: HardwareCapabilityProfile,
) -> CatalogFitAssessment:
    params_b = infer_effective_params_b(
        title=entry.title,
        description=entry.description,
        repo_id=entry.gguf_repo,
    )
    est_gb = estimate_q4_footprint_gb(params_b)
    budget = profile.inference_budget_gb

    if est_gb is None or budget <= 0:
        fit = CatalogFitLevel.UNKNOWN
        ratio = 999.0
        rationale = "Model size could not be estimated from catalog metadata."
        score = 20.0 + _tier_bonus(entry, profile)
    else:
        ratio = est_gb / budget
        fit = _fit_level_for_ratio(ratio, True)
        score = {
            CatalogFitLevel.EXCELLENT: 100.0,
            CatalogFitLevel.GOOD: 82.0,
            CatalogFitLevel.MARGINAL: 45.0,
            CatalogFitLevel.POOR: 10.0,
        }[fit]
        score -= max(0.0, (ratio - 1.0) * 25.0)
        score += _tier_bonus(entry, profile)
        if fit == CatalogFitLevel.EXCELLENT:
            rationale = f"Estimated ~{est_gb:.1f} GB Q4 fits comfortably in your ~{budget:.1f} GB budget."
        elif fit == CatalogFitLevel.GOOD:
            rationale = f"Estimated ~{est_gb:.1f} GB Q4 should run well on your hardware."
        elif fit == CatalogFitLevel.MARGINAL:
            rationale = f"Estimated ~{est_gb:.1f} GB Q4 may run slower; consider a smaller quant or model."
        else:
            rationale = f"Estimated ~{est_gb:.1f} GB Q4 exceeds your ~{budget:.1f} GB budget."

    return CatalogFitAssessment(
        catalog_id=entry.catalog_id,
        title=entry.title,
        fit_level=fit,
        score=score,
        params_b=params_b,
        estimated_q4_gb=est_gb,
        rationale=rationale,
    )


def build_catalog_recommendation_plan(
    entries: list[CatalogEntry],
    profile: HardwareCapabilityProfile | None = None,
) -> CatalogRecommendationPlan:
    if profile is None:
        profile = detect_hardware_capability_profile()

    assessments = tuple(
        sorted(
            (_score_entry(e, profile) for e in entries),
            key=lambda a: (-a.score, a.title.lower()),
        )
    )
    recommended = tuple(
        a for a in assessments if a.fit_level in (CatalogFitLevel.EXCELLENT, CatalogFitLevel.GOOD)
    )[:3]

    if recommended:
        names = ", ".join(a.title for a in recommended[:2])
        if len(recommended) > 2:
            names += f", or {recommended[2].title}"
        banner = f"For your system ({profile.summary_label}): start with {names}."
    elif profile.inference_budget_gb > 0:
        banner = (
            f"Your system ({profile.summary_label}) has a tight memory budget "
            f"(~{profile.inference_budget_gb:.1f} GB). Prefer the smallest verified models and Q4 quants."
        )
    else:
        banner = "Hardware could not be fully detected — compare file sizes before downloading."

    detail = (
        f"{profile.tier_label} capability profile. "
        f"{profile.summary_label}. "
        + (
            f"Recommended models are ranked for a ~{profile.inference_budget_gb:.1f} GB Q4-class load."
            if profile.inference_budget_gb > 0
            else "Use quant badges and file sizes to choose a download."
        )
    )

    return CatalogRecommendationPlan(
        profile=profile,
        assessments=assessments,
        recommended=recommended,
        banner_text=banner,
        detail_text=detail,
    )


def sort_entries_by_hardware_fit(
    entries: list[CatalogEntry],
    plan: CatalogRecommendationPlan,
) -> list[CatalogEntry]:
    rank = {a.catalog_id: idx for idx, a in enumerate(plan.assessments)}
    return sorted(entries, key=lambda e: (rank.get(e.catalog_id, 999), e.title.lower()))


def build_tour_model_download_body(
    entries: list[CatalogEntry] | None = None,
    profile: HardwareCapabilityProfile | None = None,
) -> str:
    """Coach-mark copy for the Local LLM setup tour Model Manager step."""
    catalog = entries if entries is not None else load_qube_verified_models()
    plan = build_catalog_recommendation_plan(catalog, profile=profile)
    intro = (
        "Open Model Manager to browse Qube Verified models on Hugging Face, "
        "download a .gguf, then return here and pick it from Select AI Model."
    )
    if not plan.recommended:
        if plan.profile.inference_budget_gb > 0:
            return (
                f"{intro}\n\nYour system ({plan.profile.summary_label}) has a modest "
                f"memory budget (~{plan.profile.inference_budget_gb:.1f} GB for a Q4-class load). "
                "Start with the smallest verified models and Q4 quants."
            )
        return intro

    if len(plan.recommended) == 1:
        picks = plan.recommended[0].title
    elif len(plan.recommended) == 2:
        picks = f"{plan.recommended[0].title} or {plan.recommended[1].title}"
    else:
        picks = (
            f"{plan.recommended[0].title}, {plan.recommended[1].title}, "
            f"or {plan.recommended[2].title}"
        )
    return (
        f"{intro}\n\nFor your system ({plan.profile.summary_label}), "
        f"good starting points are {picks}."
    )
