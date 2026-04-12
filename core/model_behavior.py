"""
Runtime model behavior classification from ablation / optional diagnostics (metadata only).

Does not change prompts or sampling — feeds execution-policy overrides in NativeLlamaEngine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from core.execution_policy import ExecutionPolicy

if TYPE_CHECKING:
    from core.prompt_ablation_harness import AblationReport


class ModelBehaviorClass(str, Enum):
    WELL_BEHAVED = "WELL_BEHAVED"
    FORCED_REASONING = "FORCED_REASONING"
    TEMPLATE_SENSITIVE = "TEMPLATE_SENSITIVE"
    STOP_TOKEN_DEPENDENT = "STOP_TOKEN_DEPENDENT"
    UNSTABLE_FIRST_TOKEN = "UNSTABLE_FIRST_TOKEN"
    UNKNOWN = "UNKNOWN"


@dataclass
class ModelBehaviorProfile:
    model_name: str
    first_token: str
    first_token_id: int | None
    leakage_detected: bool
    template_sensitivity_score: float
    stop_token_dependency_score: float
    reasoning_triggered_without_prompt: bool
    behavior_class: ModelBehaviorClass
    confidence: float


def _baseline_result(report: "AblationReport") -> Any:
    for r in report.scenario_results:
        if r.scenario == "BASELINE":
            return r
    return report.scenario_results[0] if report.scenario_results else None


def _text_starts_with_redacted(full: str) -> bool:
    t = (full or "").lstrip()
    return t.startswith("<redacted_thinking>")


def _majority_leakage(report: "AblationReport", threshold: float = 0.5) -> bool:
    rows = report.scenario_results
    if not rows:
        return False
    n = len(rows)
    k = sum(1 for r in rows if r.leakage_detected)
    return (k / float(n)) > threshold


def classify_model_behavior(
    ablation_report: Optional["AblationReport"],
    ground_truth_trace: Any = None,
    causality_report: Any = None,
    *,
    model_name: str = "",
) -> ModelBehaviorProfile:
    """
    Map ablation (and optional GT / causality) to a behavior class and confidence.

    Confidence: higher when multiple independent signals agree; lower on fallback / UNKNOWN.
    """
    if ablation_report is None or not getattr(ablation_report, "scenario_results", None):
        return ModelBehaviorProfile(
            model_name=model_name or "",
            first_token="",
            first_token_id=None,
            leakage_detected=False,
            template_sensitivity_score=0.0,
            stop_token_dependency_score=0.0,
            reasoning_triggered_without_prompt=False,
            behavior_class=ModelBehaviorClass.UNKNOWN,
            confidence=0.2,
        )

    rep = ablation_report
    base = _baseline_result(rep)
    first_20 = (getattr(base, "first_20_tokens", None) or "") if base else ""
    full_sample = (getattr(base, "full_text_sample", None) or "") if base else ""
    first_char = (getattr(base, "first_token", None) or "") if base else ""

    tpl = float(rep.template_sensitivity_score or 0.0)
    stop_dep = float(rep.stop_token_dependency_score or 0.0)
    leak_majority = _majority_leakage(rep, threshold=0.5)
    baseline_leak = bool(getattr(base, "leakage_detected", False)) if base else False

    starts_redacted = _text_starts_with_redacted(full_sample) or _text_starts_with_redacted(
        first_20
    )
    reasoning_no_prompt = starts_redacted or (
        baseline_leak and (getattr(base, "has_redacted_thinking", False) or starts_redacted)
    )

    first_token_id: Optional[int] = None
    if isinstance(ground_truth_trace, dict):
        first_token_id = ground_truth_trace.get("first_token_id")
        if first_token_id is not None:
            try:
                first_token_id = int(first_token_id)
            except (TypeError, ValueError):
                first_token_id = None

    # UNSTABLE_FIRST_TOKEN: optional signal from causality or drift without crossing template threshold
    unstable_from_causality = False
    if isinstance(causality_report, dict):
        lbl = str(causality_report.get("first_token_cause_label") or "").lower()
        unstable_from_causality = "unstable" in lbl or "diverge" in lbl

    uniq_first = (
        len({v for v in rep.first_token_drift_map.values()})
        if rep.first_token_drift_map
        else 0
    )
    unstable_drift = uniq_first >= 3 and tpl <= 0.5 and stop_dep < 0.5

    if starts_redacted:
        behavior = ModelBehaviorClass.FORCED_REASONING
    elif leak_majority:
        behavior = ModelBehaviorClass.FORCED_REASONING
    elif tpl > 0.5:
        behavior = ModelBehaviorClass.TEMPLATE_SENSITIVE
    elif stop_dep > 0.5:
        behavior = ModelBehaviorClass.STOP_TOKEN_DEPENDENT
    elif unstable_from_causality or unstable_drift:
        behavior = ModelBehaviorClass.UNSTABLE_FIRST_TOKEN
    else:
        behavior = ModelBehaviorClass.WELL_BEHAVED

    conf = 0.55
    if behavior == ModelBehaviorClass.FORCED_REASONING:
        n_sig = (1 if starts_redacted else 0) + (1 if leak_majority else 0)
        conf = 0.92 if n_sig >= 2 else 0.78 if starts_redacted or leak_majority else 0.55
    elif behavior == ModelBehaviorClass.TEMPLATE_SENSITIVE:
        conf = min(0.95, 0.55 + 0.35 * tpl)
    elif behavior == ModelBehaviorClass.STOP_TOKEN_DEPENDENT:
        conf = min(0.95, 0.5 + 0.45 * stop_dep)
    elif behavior == ModelBehaviorClass.UNSTABLE_FIRST_TOKEN:
        conf = 0.5 if unstable_drift else 0.45
    elif behavior == ModelBehaviorClass.WELL_BEHAVED:
        conf = 0.65 if not leak_majority and tpl <= 0.2 else 0.45

    display_first = (first_20[:32] if first_20 else (first_char or ""))[:64]
    return ModelBehaviorProfile(
        model_name=model_name,
        first_token=display_first,
        first_token_id=first_token_id,
        leakage_detected=bool(baseline_leak or leak_majority),
        template_sensitivity_score=tpl,
        stop_token_dependency_score=stop_dep,
        reasoning_triggered_without_prompt=bool(reasoning_no_prompt),
        behavior_class=behavior,
        confidence=round(min(1.0, max(0.0, conf)), 4),
    )


def resolve_behavior_override(profile: ModelBehaviorProfile) -> Dict[str, Any]:
    """
    Map behavior class to execution overrides (no prompt / sampling changes).

    Returns keys: force_execution_mode, enforcement_mode, strip_thinking, reason.
    """
    cls = profile.behavior_class
    if cls == ModelBehaviorClass.FORCED_REASONING:
        return {
            "force_execution_mode": "direct",
            "enforcement_mode": "hard",
            "strip_thinking": True,
            "reason": "Model forces reasoning / leakage in ablation probe — direct + hard strip.",
        }
    if cls == ModelBehaviorClass.TEMPLATE_SENSITIVE:
        return {
            "force_execution_mode": None,
            "enforcement_mode": "soft",
            "strip_thinking": False,
            "reason": "High template sensitivity — prefer soft enforcement.",
        }
    if cls == ModelBehaviorClass.STOP_TOKEN_DEPENDENT:
        return {
            "force_execution_mode": None,
            "enforcement_mode": None,
            "strip_thinking": False,
            "reason": "Stop-token dependent output; no execution-mode override (stops handled elsewhere).",
        }
    if cls == ModelBehaviorClass.UNSTABLE_FIRST_TOKEN:
        return {
            "force_execution_mode": None,
            "enforcement_mode": "soft",
            "strip_thinking": False,
            "reason": "Unstable first-token behavior — soft enforcement only.",
        }
    if cls == ModelBehaviorClass.UNKNOWN:
        return {
            "force_execution_mode": None,
            "enforcement_mode": None,
            "strip_thinking": False,
            "reason": "Insufficient ablation data — no override.",
        }
    # WELL_BEHAVED
    return {
        "force_execution_mode": None,
        "enforcement_mode": None,
        "strip_thinking": False,
        "reason": "Well-behaved in probe — no override.",
    }


def apply_behavior_override_to_policy(
    pol: ExecutionPolicy,
    override: Optional[Dict[str, Any]],
) -> ExecutionPolicy:
    """Apply non-sampling execution fields; keeps policy coherent for stream/UI."""
    if not override:
        return pol
    p = pol
    fe = override.get("force_execution_mode")
    en = override.get("enforcement_mode")
    st = bool(override.get("strip_thinking"))

    if fe is not None:
        # ExecutionMode is Literal — trust caller
        p = replace(p, execution_mode=fe)  # type: ignore[arg-type]
    if en is not None:
        p = replace(p, enforcement_mode=en)  # type: ignore[arg-type]
    if st:
        p = replace(p, strip_thinking_output=True, tts_strip_thinking=True)

    if fe == "direct":
        p = replace(
            p,
            allow_thinking_tokens=False,
            ui_display_thinking=False,
        )
    return p


def override_materially_changes_policy(
    base: ExecutionPolicy,
    override: Optional[Dict[str, Any]],
) -> bool:
    if not override:
        return False
    applied = apply_behavior_override_to_policy(base, override)
    return (
        applied.execution_mode != base.execution_mode
        or applied.enforcement_mode != base.enforcement_mode
        or applied.strip_thinking_output != base.strip_thinking_output
        or applied.allow_thinking_tokens != base.allow_thinking_tokens
        or applied.ui_display_thinking != base.ui_display_thinking
    )


def behavior_profile_log_event(
    *,
    model: str,
    profile: ModelBehaviorProfile,
    override: Dict[str, Any],
    override_active: bool,
) -> str:
    """Single JSON line for Qube.NativeLLM.Debug."""
    payload = {
        "event": "llm_model_behavior_profile",
        "model": model,
        "behavior_class": profile.behavior_class.value,
        "confidence": profile.confidence,
        "first_token": profile.first_token,
        "override_applied": bool(override_active),
        "override": dict(override),
    }
    return json.dumps(payload, ensure_ascii=False)
