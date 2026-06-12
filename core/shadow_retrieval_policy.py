"""
Shadow continuous retrieval policy for LLMWorker (observational only).

Replaces binary recall_fusion observation with a continuous propensity model
in parallel to baseline execution. Does NOT modify routing or retrieval.
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("Qube.ShadowRetrievalPolicy")

_SHADOW_POLICY_ENV = "QUBE_SHADOW_RETRIEVAL_POLICY"
_DEFAULT_T_NONE = 0.30
_DEFAULT_DELTA = 0.08
_OVERLAP_PENALTY = 0.5
_SIGMOID_A1 = 5.0


def shadow_retrieval_policy_enabled() -> bool:
    """Shadow policy observation is on by default; set env to 0 to disable."""
    return os.environ.get(_SHADOW_POLICY_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _normalize_route(route: str) -> str:
    r = str(route or "none").strip().lower()
    aliases = {"internet": "web", "chat": "none"}
    return aliases.get(r, r)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(x: float, a: float = _SIGMOID_A1) -> float:
    z = max(-20.0, min(20.0, a * x))
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class PolicyWeights:
    w1: float = 0.30
    w2: float = 0.25
    w3: float = 0.20
    w4: float = 0.15
    w5: float = 0.10


@dataclass(frozen=True)
class PolicyThresholds:
    t_none: float = _DEFAULT_T_NONE
    delta: float = _DEFAULT_DELTA
    t_semantic: Optional[float] = None
    t_contextual: Optional[float] = None


@dataclass(frozen=True)
class PropensityAxes:
    """Independent semantic (router) vs contextual (discourse) retrieval axes."""

    semantic_raw: float
    contextual_raw: float
    semantic_norm: float
    contextual_norm: float
    combined: float


@dataclass
class ShadowRetrievalState:
    """Inputs available at LLMWorker routing boundary."""

    baseline_route: str
    decision: dict[str, Any]
    prompt: str = ""
    chat_score: float = 0.0
    confidence_margin: float = 0.0
    top_score: float = 0.0
    second_best_score: float = 0.0
    follow_up_strength: float = 0.0
    discourse_continuation: float = 0.0
    memory_enabled: bool = True
    rag_enabled: bool = True
    baseline_recall_fusion: bool = False
    weights: PolicyWeights = field(default_factory=PolicyWeights)
    thresholds: PolicyThresholds = field(default_factory=PolicyThresholds)


def detect_baseline_recall_fusion(state: ShadowRetrievalState) -> bool:
    """Mirror baseline recall-fusion detection without mutating routing."""
    if state.baseline_recall_fusion:
        return True
    if bool(state.decision.get("recall_fusion")):
        return True
    try:
        from core.memory_filters import detect_recall_intent

        if (
            _normalize_route(state.baseline_route) == "hybrid"
            and detect_recall_intent((state.prompt or "").lower().strip())
        ):
            return True
    except ImportError:
        pass
    return False


def _retrieval_affinities(
    route: str,
    *,
    memory_enabled: bool,
    rag_enabled: bool,
) -> tuple[float, float]:
    route_n = _normalize_route(route)
    memory_aff = 0.35 if memory_enabled else 0.0
    rag_aff = 0.35 if rag_enabled else 0.0
    if route_n == "memory":
        memory_aff = 0.85
    elif route_n == "rag":
        rag_aff = 0.85
    elif route_n == "hybrid":
        memory_aff = 0.75 if memory_enabled else 0.0
        rag_aff = 0.75 if rag_enabled else 0.0
    elif route_n == "web":
        memory_aff = 0.20
        rag_aff = 0.20
    return _clamp(memory_aff), _clamp(rag_aff)


def _semantic_weight_sum(weights: PolicyWeights) -> float:
    return weights.w1 + weights.w2 + weights.w3


def _contextual_weight_sum(weights: PolicyWeights) -> float:
    return weights.w4 + weights.w5


def decompose_propensity_axes(state: ShadowRetrievalState) -> PropensityAxes:
    """
    Split propensity into independent axes.

    Semantic: router margin, chat avoidance, confidence margin (w1–w3).
    Contextual: follow-up strength and discourse continuation (w4–w5).
    Normalized axes are in [0, 1] within each component weight budget.
    """
    w = state.weights
    sep = state.top_score - state.second_best_score
    semantic_raw = (
        w.w1 * _sigmoid(sep)
        + w.w2 * _clamp(1.0 - state.chat_score)
        + w.w3 * _clamp(1.0 - state.confidence_margin)
    )
    contextual_raw = (
        w.w4 * _clamp(state.follow_up_strength)
        + w.w5 * _clamp(state.discourse_continuation)
    )
    sem_max = _semantic_weight_sum(w) or 1.0
    ctx_max = _contextual_weight_sum(w) or 1.0
    combined = _clamp(semantic_raw + contextual_raw)
    return PropensityAxes(
        semantic_raw=round(semantic_raw, 4),
        contextual_raw=round(contextual_raw, 4),
        semantic_norm=round(_clamp(semantic_raw / sem_max), 4),
        contextual_norm=round(_clamp(contextual_raw / ctx_max), 4),
        combined=round(combined, 4),
    )


def _propensity_score(state: ShadowRetrievalState) -> float:
    return decompose_propensity_axes(state).combined


def axes_activate_retrieval(
    axes: PropensityAxes,
    *,
    t_semantic: float,
    t_contextual: float,
) -> bool:
    """
    2D OR-gate with axis disable at T=0.

    T_semantic=0 disables the semantic gate; T_contextual=0 disables contextual.
    When both are zero, retrieval is permissive (legacy 1D low-threshold behavior).
    """
    sem_gate = t_semantic > 0.0 and axes.semantic_norm >= t_semantic
    ctx_gate = t_contextual > 0.0 and axes.contextual_norm >= t_contextual
    if t_semantic <= 0.0 and t_contextual <= 0.0:
        return True
    if t_semantic <= 0.0:
        return ctx_gate
    if t_contextual <= 0.0:
        return sem_gate
    return sem_gate or ctx_gate


def _uses_2d_thresholds(thresholds: PolicyThresholds) -> bool:
    return thresholds.t_semantic is not None or thresholds.t_contextual is not None


def _retrieval_activated(
    propensity: float,
    axes: PropensityAxes,
    *,
    thresholds: PolicyThresholds,
) -> bool:
    if _uses_2d_thresholds(thresholds):
        t_sem = thresholds.t_semantic if thresholds.t_semantic is not None else 0.0
        t_ctx = thresholds.t_contextual if thresholds.t_contextual is not None else 0.0
        return axes_activate_retrieval(axes, t_semantic=t_sem, t_contextual=t_ctx)
    return propensity >= thresholds.t_none


def _probabilities(
    propensity: float,
    *,
    memory_affinity: float,
    rag_affinity: float,
) -> tuple[float, float, float]:
    p_memory = _clamp(propensity * memory_affinity)
    p_rag = _clamp(propensity * rag_affinity)
    overlap = p_memory * p_rag * _OVERLAP_PENALTY
    p_hybrid = _clamp(p_memory + p_rag - overlap)
    return round(p_memory, 4), round(p_rag, 4), round(p_hybrid, 4)


def _shadow_decision(
    propensity: float,
    p_memory: float,
    p_rag: float,
    *,
    thresholds: PolicyThresholds,
    fallback_route: str,
    axes: PropensityAxes | None = None,
) -> str:
    axis_obj = axes or PropensityAxes(
        semantic_raw=propensity,
        contextual_raw=0.0,
        semantic_norm=propensity,
        contextual_norm=0.0,
        combined=propensity,
    )
    if not _retrieval_activated(propensity, axis_obj, thresholds=thresholds):
        return "none"
    if p_memory > p_rag + thresholds.delta:
        return "memory"
    if p_rag > p_memory + thresholds.delta:
        return "rag"
    if p_memory > 0.05 or p_rag > 0.05:
        return "hybrid"
    return _normalize_route(fallback_route)


def _is_retrieval_route(route: str) -> bool:
    return _normalize_route(route) in {"memory", "rag", "hybrid", "web"}


def compute_retrieval_policy(state: ShadowRetrievalState) -> dict[str, Any]:
    """
    Compute shadow continuous retrieval policy for a single turn.

    Returns policy dict suitable for telemetry and eval aggregation.
    """
    baseline_fusion = detect_baseline_recall_fusion(state)
    axes = decompose_propensity_axes(state)
    propensity = axes.combined
    mem_aff, rag_aff = _retrieval_affinities(
        state.baseline_route,
        memory_enabled=state.memory_enabled,
        rag_enabled=state.rag_enabled,
    )
    p_memory, p_rag, p_hybrid = _probabilities(
        propensity,
        memory_affinity=mem_aff,
        rag_affinity=rag_aff,
    )
    shadow_route = _shadow_decision(
        propensity,
        p_memory,
        p_rag,
        thresholds=state.thresholds,
        fallback_route=state.baseline_route,
        axes=axes,
    )
    baseline_norm = _normalize_route(state.baseline_route)
    shadow_norm = _normalize_route(shadow_route)
    baseline_retrieval = _is_retrieval_route(baseline_norm)
    shadow_retrieval = _is_retrieval_route(shadow_norm)

    return {
        "retrieval_propensity_score": propensity,
        "semantic_axis_score": axes.semantic_norm,
        "contextual_axis_score": axes.contextual_norm,
        "semantic_axis_raw": axes.semantic_raw,
        "contextual_axis_raw": axes.contextual_raw,
        "P_memory": p_memory,
        "P_rag": p_rag,
        "P_hybrid": p_hybrid,
        "shadow_decision": shadow_norm,
        "baseline_recall_fusion": baseline_fusion,
        "delta_vs_baseline": {
            "route_change": baseline_norm != shadow_norm,
            "retrieval_change": baseline_retrieval != shadow_retrieval,
        },
    }


def build_shadow_state_from_worker(
    *,
    execution_route: str,
    decision: dict[str, Any],
    prompt: str,
    follow_up: Any = None,
    discourse_state: Any = None,
    memory_enabled: bool = True,
    rag_enabled: bool = True,
) -> ShadowRetrievalState:
    """Construct ``ShadowRetrievalState`` from LLMWorker routing context."""
    follow_strength = 0.0
    if follow_up is not None and getattr(follow_up, "active", False):
        follow_strength = float(getattr(follow_up, "confidence", 0.0) or 0.0)

    discourse_signal = 0.0
    if discourse_state is not None and getattr(discourse_state, "active_topic", None):
        discourse_signal = 0.5
        if follow_up is not None and getattr(follow_up, "active", False):
            discourse_signal = 0.7

    return ShadowRetrievalState(
        baseline_route=execution_route,
        decision=dict(decision or {}),
        prompt=prompt,
        chat_score=float(decision.get("chat_score") or 0.0),
        confidence_margin=float(decision.get("confidence_margin") or 0.0),
        top_score=float(decision.get("top_score") or 0.0),
        second_best_score=float(decision.get("second_best_score") or 0.0),
        follow_up_strength=follow_strength,
        discourse_continuation=discourse_signal,
        memory_enabled=memory_enabled,
        rag_enabled=rag_enabled,
        baseline_recall_fusion=bool(decision.get("recall_fusion")),
    )


class ShadowRetrievalPolicyTelemetry:
    """Rolling aggregator for shadow policy divergence metrics."""

    def __init__(self, max_samples: int = 400) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_samples)

    def record(
        self,
        *,
        baseline_route: str,
        shadow_policy: dict[str, Any],
        prompt: str = "",
        category: str = "",
    ) -> None:
        baseline_norm = _normalize_route(baseline_route)
        shadow_norm = shadow_policy.get("shadow_decision", "none")
        event = {
            "ts": time.time(),
            "baseline_route": baseline_norm,
            "shadow_route": shadow_norm,
            "retrieval_propensity_score": shadow_policy.get("retrieval_propensity_score"),
            "baseline_recall_fusion": shadow_policy.get("baseline_recall_fusion"),
            "route_divergence": baseline_norm != shadow_norm,
            "retrieval_divergence": shadow_policy.get("delta_vs_baseline", {}).get(
                "retrieval_change", False
            ),
            "prompt_excerpt": (prompt or "")[:120],
            "category": category,
            "shadow_suppresses_baseline_retrieval": (
                _is_retrieval_route(baseline_norm) and shadow_norm == "none"
            ),
            "shadow_enables_retrieval": (
                baseline_norm == "none" and _is_retrieval_route(shadow_norm)
            ),
        }
        self._events.append(event)

    def summarize(self) -> dict[str, Any]:
        if not self._events:
            return {}

        n = len(self._events)
        propensity_scores = [
            float(e["retrieval_propensity_score"])
            for e in self._events
            if e.get("retrieval_propensity_score") is not None
        ]
        diverged = sum(1 for e in self._events if e.get("route_divergence"))
        fusion_baseline = sum(1 for e in self._events if e.get("baseline_recall_fusion"))
        fusion_shadow_none = sum(
            1
            for e in self._events
            if e.get("baseline_recall_fusion") and e.get("shadow_route") == "none"
        )
        hybrid_baseline = sum(1 for e in self._events if e.get("baseline_route") == "hybrid")
        hybrid_shadow_not = sum(
            1
            for e in self._events
            if e.get("baseline_route") == "hybrid" and e.get("shadow_route") != "hybrid"
        )
        none_baseline = sum(1 for e in self._events if e.get("baseline_route") == "none")
        none_shadow_retrieval = sum(
            1
            for e in self._events
            if e.get("baseline_route") == "none" and _is_retrieval_route(
                str(e.get("shadow_route") or "none")
            )
        )
        suppress_regression = sum(
            1 for e in self._events if e.get("shadow_suppresses_baseline_retrieval")
        )
        shadow_improves = sum(
            1
            for e in self._events
            if e.get("baseline_recall_fusion") and e.get("shadow_route") == "none"
        )

        return {
            "samples": n,
            "avg_propensity_score": round(
                statistics.mean(propensity_scores) if propensity_scores else 0.0, 4
            ),
            "divergence_rate": round(diverged / n, 4),
            "recall_fusion_eliminated_rate": round(
                (fusion_baseline - fusion_shadow_none) / fusion_baseline
                if fusion_baseline
                else 0.0,
                4,
            ),
            "shadow_replacement_rate": round(diverged / n, 4),
            "hybrid_stability_gain": round(
                hybrid_shadow_not / hybrid_baseline if hybrid_baseline else 0.0, 4
            ),
            "hybrid_reduction_rate": round(
                hybrid_shadow_not / hybrid_baseline if hybrid_baseline else 0.0, 4
            ),
            "none_reduction_rate": round(
                none_shadow_retrieval / none_baseline if none_baseline else 0.0, 4
            ),
            "retrieval_coverage_delta": round(
                (none_shadow_retrieval - suppress_regression) / n, 4
            ),
            "retrieval_stability_gain_estimate": round(
                shadow_improves / fusion_baseline if fusion_baseline else 0.0, 4
            ),
            "regression_suppression_count": suppress_regression,
            "stability_improvement_count": shadow_improves,
            "best_thresholds": {
                "T_none": _DEFAULT_T_NONE,
                "delta": _DEFAULT_DELTA,
                "weights": asdict(PolicyWeights()),
            },
        }


_global_telemetry: Optional[ShadowRetrievalPolicyTelemetry] = None


def get_shadow_retrieval_telemetry() -> ShadowRetrievalPolicyTelemetry:
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = ShadowRetrievalPolicyTelemetry()
    return _global_telemetry
