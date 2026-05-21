"""
L3 heuristic model router: selects best candidate model profile per user query.

Advisory at inference boundary: does not load/swap GGUF; NativeLlamaEngine logs
decision and updates registry feedback. PromptContract remains unchanged.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from core.model_performance_store import ModelPerformanceStore

_EMA_ALPHA_QUALITY = 0.15
_EMA_ALPHA_LATENCY = 0.10
_PERF_HINT_MAX_ABS = 0.06
_LOW_QUALITY_THRESHOLD = 0.40


@dataclass
class ModelProfile:
    name: str
    context_length: int
    strengths: list[str]  # e.g. "reasoning", "coding", "chat", "summarization"
    weaknesses: list[str]
    avg_quality_score: float
    avg_latency: float
    usage_count: int
    model_path: Optional[str] = None


# --- Registry (in-process) ---
_REGISTRY: dict[str, ModelProfile] = {}
_PERFORMANCE_STORE: Optional[ModelPerformanceStore] = None


@dataclass
class RoutingDecision:
    selected_model: str
    confidence: float
    reasoning: list[str]
    task: str = "general_chat"
    scores: dict[str, float] = field(default_factory=dict)


def _get_performance_store() -> ModelPerformanceStore:
    global _PERFORMANCE_STORE
    if _PERFORMANCE_STORE is None:
        _PERFORMANCE_STORE = ModelPerformanceStore()
    return _PERFORMANCE_STORE


def set_performance_store_for_tests(store: Optional[ModelPerformanceStore]) -> None:
    """Test helper to inject deterministic store state."""
    global _PERFORMANCE_STORE
    _PERFORMANCE_STORE = store


def extract_last_user_query(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if str(m.get("role", "")).lower() != "user":
            continue
        txt = str(m.get("content") or "")
        if "USER QUERY:" in txt:
            txt = txt.rsplit("USER QUERY:", 1)[-1].strip()
        return txt
    return ""


def register_model_profiles(profiles: list[ModelProfile]) -> None:
    for p in profiles:
        _REGISTRY[p.name] = p


def clear_router_registry() -> None:
    """Test helper."""
    _REGISTRY.clear()


def get_registry_models() -> list[ModelProfile]:
    return list(_REGISTRY.values())


def _infer_strengths_from_basename(basename: str) -> list[str]:
    low = basename.lower()
    out: list[str] = []
    if any(k in low for k in ("code", "coder", "dev", "starcoder", "deepseek-coder")):
        out.append("coding")
    if any(k in low for k in ("reason", "r1", "think", "qwq", "o1", "o3", "nemotron")):
        out.append("reasoning")
    if any(k in low for k in ("summar", "mini", "small", "tiny", "0.5b", "1b", "3b")):
        out.append("summarization")
    if not out or "chat" not in out:
        out.append("chat")
    return list(dict.fromkeys(out))


def upsert_profile_from_loaded_model(
    *,
    model_path: str,
    display_name: str,
    context_length: int,
) -> None:
    """Register or refresh the currently loaded native model in the router registry."""
    name = (display_name or "").strip() or os.path.basename(model_path) or "unknown"
    strengths = _infer_strengths_from_basename(name)
    if name in _REGISTRY:
        p = _REGISTRY[name]
        p.model_path = model_path
        p.context_length = int(context_length)
        p.strengths = strengths or p.strengths
        return
    _REGISTRY[name] = ModelProfile(
        name=name,
        context_length=int(context_length),
        strengths=strengths,
        weaknesses=[],
        avg_quality_score=0.5,
        avg_latency=5.0,
        usage_count=0,
        model_path=model_path,
    )


def record_inference_feedback(
    model_name: str,
    quality_score: Optional[float],
    latency_ms: Optional[float] = None,
) -> None:
    """Update moving averages after a turn (L2 quality optional)."""
    key = (model_name or "").strip()
    if not key or key not in _REGISTRY:
        return
    p = _REGISTRY[key]
    p.usage_count += 1
    if quality_score is not None:
        try:
            q = float(quality_score)
        except (TypeError, ValueError):
            q = None
        if q is not None:
            if p.usage_count <= 1:
                p.avg_quality_score = q
            else:
                p.avg_quality_score = (1.0 - _EMA_ALPHA_QUALITY) * p.avg_quality_score + _EMA_ALPHA_QUALITY * q
    if latency_ms is not None:
        try:
            lat = float(latency_ms)
        except (TypeError, ValueError):
            lat = None
        if lat is not None:
            if p.usage_count <= 1:
                p.avg_latency = lat
            else:
                p.avg_latency = (1.0 - _EMA_ALPHA_LATENCY) * p.avg_latency + _EMA_ALPHA_LATENCY * lat


def _classify_task(user_query: str, task_hint: Optional[str]) -> tuple[str, list[str]]:
    if task_hint:
        t = task_hint.strip().lower()
        if t in ("coding", "reasoning", "summarization", "general_chat"):
            return t, [f"task_hint={t}"]
    q = (user_query or "").lower()
    notes: list[str] = []
    if any(k in q for k in ("def ", "import ", "```", "class ", "exception", "traceback", "typescript", "javascript")):
        return "coding", ["keyword: code/def/import"]
    if any(k in q for k in ("prove", "why must", "formal", "theorem", "contradiction", "step by step logic")):
        return "reasoning", ["keyword: formal reasoning"]
    if any(k in q for k in ("summarize", "tl;dr", "tldr", "in one paragraph", "brief summary")):
        return "summarization", ["keyword: summarize"]
    if len(q) > 200 and ("?" in user_query or "explain" in q):
        return "reasoning", ["long analytical query"]
    return "general_chat", ["default"]


def _latency_sensitive(user_query: str) -> bool:
    q = (user_query or "").lower()
    return any(k in q for k in ("quick", "fast", "asap", "short reply", "one sentence", "briefly"))


def _task_strength_bonus(profile: ModelProfile, task: str) -> float:
    s = {x.lower() for x in (profile.strengths or [])}
    if task == "coding" and "coding" in s:
        return 0.35
    if task == "reasoning" and "reasoning" in s:
        return 0.35
    if task == "summarization" and "summarization" in s:
        return 0.30
    if task == "general_chat" and "chat" in s:
        return 0.15
    return 0.0


def _weakness_penalty(profile: ModelProfile, task: str) -> float:
    w = {x.lower() for x in (profile.weaknesses or [])}
    if task == "coding" and "coding" in w:
        return 0.25
    if task == "reasoning" and "reasoning" in w:
        return 0.25
    return 0.0


def _score_model(
    profile: ModelProfile,
    task: str,
    *,
    context_chars: int,
    latency_sensitive: bool,
) -> tuple[float, list[str]]:
    parts: list[str] = []
    score = 0.0
    score += _task_strength_bonus(profile, task)
    parts.append(f"task_match={_task_strength_bonus(profile, task):.2f}")
    score -= _weakness_penalty(profile, task)
    parts.append(f"weakness_penalty={_weakness_penalty(profile, task):.2f}")

    qscore = float(profile.avg_quality_score or 0.5)
    score += 0.25 * max(0.0, min(1.0, qscore))
    parts.append(f"quality={0.25 * max(0.0, min(1.0, qscore)):.2f}")

    lat = float(profile.avg_latency or 5.0)
    lat_pen = min(0.35, max(0.0, (lat - 2.0) / 40.0))
    score -= lat_pen
    parts.append(f"latency_penalty=-{lat_pen:.2f}")
    if latency_sensitive:
        extra = min(0.2, max(0.0, lat / 60.0))
        score -= extra
        parts.append(f"latency_sensitive_extra=-{extra:.2f}")

    need_ctx = max(len(str(context_chars)), 2000)
    if profile.context_length >= need_ctx:
        score += 0.1
        parts.append("context_ok=+0.10")
    else:
        gap = (need_ctx - profile.context_length) / max(need_ctx, 1)
        score -= 0.15 * min(1.0, gap)
        parts.append(f"context_tight=-{0.15 * min(1.0, gap):.2f}")

    usage = int(profile.usage_count or 0)
    smooth = min(0.05, 0.01 * math.log(1 + max(0, usage)))
    score += smooth
    parts.append(f"usage_smooth=+{smooth:.3f}")

    # Optional cross-model telemetry hint: bounded and never decisive on its own.
    try:
        perf_hint = _get_performance_store().get(profile.name)
    except Exception:
        perf_hint = None
    if perf_hint is not None:
        perf_delta = 0.0
        if perf_hint.structural_failure_rate > 0.6:
            perf_delta -= 0.05
            parts.append("perf_unreliable=-0.05")
        if perf_hint.avg_response_quality < _LOW_QUALITY_THRESHOLD:
            perf_delta -= 0.03
            parts.append("perf_low_quality=-0.03")
        if (
            perf_hint.structural_failure_rate <= 0.2
            and perf_hint.avg_response_quality >= 0.75
        ):
            perf_delta += 0.02
            parts.append("perf_stable_bonus=+0.02")
        perf_delta = max(-_PERF_HINT_MAX_ABS, min(_PERF_HINT_MAX_ABS, perf_delta))
        if perf_delta != 0.0:
            score += perf_delta
            parts.append(f"perf_hint={perf_delta:+.3f}")
        parts.append(
            "perf_snapshot="
            f"(fail={perf_hint.structural_failure_rate:.2f},q={perf_hint.avg_response_quality:.2f})"
        )

    return score, parts


def route_model(
    user_query: str,
    available_models: list[ModelProfile],
    context: Optional[str] = None,
    task_hint: Optional[str] = None,
) -> RoutingDecision:
    if not available_models:
        return RoutingDecision(
            selected_model="unknown",
            confidence=0.0,
            reasoning=["no profiles in registry"],
            task="general_chat",
            scores={},
        )

    task, task_notes = _classify_task(user_query, task_hint)
    ctx_len = len(context or "")
    latency_sensitive = _latency_sensitive(user_query)

    scored: list[tuple[ModelProfile, float, list[str]]] = []
    for m in available_models:
        s, parts = _score_model(m, task, context_chars=ctx_len, latency_sensitive=latency_sensitive)
        scored.append((m, s, parts))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_p, best_s, best_parts = scored[0]
    second_s = scored[1][1] if len(scored) > 1 else best_s - 0.5

    confidence = (best_s - second_s) / max(best_s, 0.01) if best_s > 0 else 0.0
    confidence = max(0.0, min(1.0, confidence))

    reasoning: list[str] = [
        f"matched_task={task}",
        *task_notes,
        *best_parts,
        f"total_score={best_s:.3f}",
    ]
    if len(scored) > 1:
        reasoning.append(f"runner_up={scored[1][0].name} score={scored[1][1]:.3f}")

    scores_map = {p.name: round(s, 4) for p, s, _ in scored}
    return RoutingDecision(
        selected_model=best_p.name,
        confidence=round(confidence, 4),
        reasoning=reasoning,
        task=task,
        scores=scores_map,
    )
