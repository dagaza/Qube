"""
Analytical counterfactual simulations over completed native inference (no model calls).

Replays structural validation on hypothetical prompt/stop/format tweaks — one intervention
at a time — and estimates risk / cause-label shifts. Deterministic.

Enable with QUBE_LLM_COUNTERFACTUAL=1 (post-inference only; see NativeLlamaEngine).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

from core.llm_execution_causality import (
    _compute_influence_scores,
    _infer_first_token_cause_label,
)
from core.native_token_trace import classify_first_token_piece
from core.prompt_integrity_validator import (
    ParityReport,
    PromptValidationResult,
    compute_parity_score,
    load_lm_studio_reference_from_env,
    native_snapshot_for_parity,
    validate_chat_inference,
)

logger = logging.getLogger("Qube.NativeLLM.Debug")


def counterfactual_report_enabled() -> bool:
    return os.environ.get("QUBE_LLM_COUNTERFACTUAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class LLMCounterfactualScenario:
    """Representative state after a single analytical intervention (for logging)."""

    modified_system_prompt: str
    modified_stop_tokens: list[str]
    modified_chat_format: str
    modified_assistant_anchor: str


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _live_vs_gt_score(
    live_token_ids: Optional[list[int]],
    ground_truth_token_ids: Optional[list[int]],
) -> Optional[float]:
    if not live_token_ids or not ground_truth_token_ids:
        return None
    from core.native_sampler_gt import sequence_prefix_match_score

    cap = min(32, len(ground_truth_token_ids), len(live_token_ids))
    return sequence_prefix_match_score(
        live_token_ids[:cap], ground_truth_token_ids[:cap]
    )


def _structural_risk(pv: PromptValidationResult) -> float:
    """Deterministic scalar — higher = worse structural posture."""
    s = 0.0
    if not pv.assistant_anchor_present:
        s += 1.0
    if pv.stop_tokens_suspicious:
        s += 0.55
    if not pv.eos_token_present:
        s += 0.45
    s += min(1.2, len(pv.risk_flags) * 0.09)
    if pv.retrieval_injection_risk:
        s += 0.25
    return round(min(4.0, s), 4)


def _synthetic_assistant_anchor_fragment(chat_format: str) -> str:
    c = (chat_format or "").lower()
    if "llama-3" in c or "start_header_id" in c:
        return "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    if "chatml" in c or "im_start" in c or "chat_template" in c or "jinja" in c:
        return "\n<|im_start|>assistant\n"
    if "mistral" in c or c == "llama-2":
        return "\n[/INST]\n"
    return "\n\n### Assistant\n"


def _system_tightening_suffix() -> str:
    return (
        "\n\n[System: Respond only as Assistant; do not emit User:/Assistant: "
        "role headers or repeat prior turns.]\n"
    )


def _expand_stop_tokens(
    merged: list[str], eos_token_str: str
) -> tuple[list[str], list[str]]:
    """Returns (new_stops, list of added sentinel labels for transparency)."""
    out: list[str] = []
    seen: set[str] = set()
    added: list[str] = []
    for s in merged:
        if s not in seen:
            seen.add(s)
            out.append(s)
    if eos_token_str and not any(eos_token_str in x for x in out):
        out.append(eos_token_str)
        added.append("eos_string_in_stops")
    for guard in ("\n\nUser:", "\n\nUSER:", "<|im_start|>user"):
        if not any(guard in x for x in out):
            out.append(guard)
            added.append(f"guard:{guard[:12]}")
    return out, added


def _parity_for_state(
    *,
    rendered_prompt: str,
    chat_format: str,
    stop_tokens: list[str],
    messages: list[dict],
    ref: Optional[dict[str, Any]],
) -> Optional[ParityReport]:
    if not ref:
        return None
    snap = native_snapshot_for_parity(
        rendered_prompt=rendered_prompt,
        chat_format=chat_format,
        stop_tokens=stop_tokens,
        messages=messages,
    )
    return compute_parity_score(snap, ref)


def _expected_shift_text(
    scenario_id: str,
    base_ft_class: str,
    baseline_cause: str,
    new_cause: str,
) -> str:
    if new_cause != baseline_cause and new_cause == "expected_generation":
        return "shifts_cause_toward_expected_generation"
    if scenario_id == "strong_assistant_anchor":
        if base_ft_class == "role_continuation":
            return "removes_role_continuation_driver"
        return "strengthens_assistant_slot_boundary"
    if scenario_id == "expanded_stop_set":
        return "reduces_meta_echo_and_premature_cutoff_risk"
    if scenario_id == "system_instruction_tightening":
        if base_ft_class == "instruction_echo":
            return "reduces_instruction_echo_opener_likelihood"
        return "clarifies_single_assistant_voice"
    if scenario_id == "aligned_reference_chat_format":
        return "aligns_template_family_with_lm_studio_reference"
    return "structural_posture_change"


def _confidence_from_reduction(reduction: float, baseline_risk: float, new_risk: float) -> float:
    if baseline_risk <= 0.001:
        return 0.35
    gain = _clamp01((baseline_risk - new_risk) / baseline_risk)
    return round(_clamp01(0.45 + 0.45 * gain + 0.1 * reduction), 4)


def _ft_class_observed(
    llama: Any,
    assistant_text: str,
    ground_truth_token_ids: Optional[list[int]],
    prompt_tokens: Optional[list[int]],
) -> str:
    if (
        llama is not None
        and ground_truth_token_ids
        and prompt_tokens is not None
    ):
        from core.native_sampler_gt import detokenize_sampler_token_ids

        pieces = detokenize_sampler_token_ids(
            llama, prompt_tokens, ground_truth_token_ids[:1]
        )
        piece = pieces[0] if pieces else ""
        return classify_first_token_piece(piece)
    if llama is not None and assistant_text:
        from core.native_token_trace import extract_first_n_generation_tokens as _ext

        _, pieces, _ = _ext(llama, assistant_text, 1)
        piece = pieces[0] if pieces else ""
        return classify_first_token_piece(piece)
    return "unknown"


def maybe_emit_counterfactual_simulations(
    llama: Any,
    *,
    rendered_prompt: Optional[str],
    messages: list[dict],
    chat_format: str,
    merged_stop_tokens: list[str],
    eos_token_str: str,
    model_metadata: dict[str, Any],
    parity_baseline: Optional[ParityReport],
    assistant_text: str,
    ground_truth_token_ids: Optional[list[int]],
    prompt_tokens: Optional[list[int]],
    live_token_ids: Optional[list[int]],
) -> None:
    """Emit one JSON line per scenario (llm_counterfactual_simulation)."""
    if not counterfactual_report_enabled():
        return
    p0 = rendered_prompt or ""
    if not p0.strip():
        return

    live_vs_gt = _live_vs_gt_score(live_token_ids, ground_truth_token_ids)
    ft_class = _ft_class_observed(
        llama, assistant_text, ground_truth_token_ids, prompt_tokens
    )

    pv0 = validate_chat_inference(
        rendered_prompt=p0,
        messages=messages,
        chat_format=chat_format,
        merged_stop_tokens=list(merged_stop_tokens),
        eos_token_str=eos_token_str,
        model_metadata=model_metadata,
        reconstruction_ok=True,
    )
    cause0 = _infer_first_token_cause_label(
        pv0, ft_class, live_vs_gt, parity_baseline
    )
    base_risk = _structural_risk(pv0)

    ref = load_lm_studio_reference_from_env()

    scenarios: list[tuple[str, Any]] = []

    # 1) Strong assistant anchor (only if currently weak)
    if not pv0.assistant_anchor_present:
        anchor_frag = _synthetic_assistant_anchor_fragment(chat_format)
        p1 = p0 + anchor_frag
        pv1 = validate_chat_inference(
            rendered_prompt=p1,
            messages=messages,
            chat_format=chat_format,
            merged_stop_tokens=list(merged_stop_tokens),
            eos_token_str=eos_token_str,
            model_metadata=model_metadata,
            reconstruction_ok=True,
        )
        par1 = _parity_for_state(
            rendered_prompt=p1,
            chat_format=chat_format,
            stop_tokens=list(merged_stop_tokens),
            messages=messages,
            ref=ref,
        )
        scenarios.append(
            (
                "strong_assistant_anchor",
                {
                    "pv": pv1,
                    "parity": par1 if par1 is not None else parity_baseline,
                    "model": LLMCounterfactualScenario(
                        modified_system_prompt="(unchanged)",
                        modified_stop_tokens=list(merged_stop_tokens),
                        modified_chat_format=chat_format,
                        modified_assistant_anchor=anchor_frag.strip()[:200],
                    ),
                },
            )
        )

    # 2) Expanded stop set
    new_stops, _ = _expand_stop_tokens(list(merged_stop_tokens), eos_token_str)
    if new_stops != list(merged_stop_tokens):
        pv2 = validate_chat_inference(
            rendered_prompt=p0,
            messages=messages,
            chat_format=chat_format,
            merged_stop_tokens=new_stops,
            eos_token_str=eos_token_str,
            model_metadata=model_metadata,
            reconstruction_ok=True,
        )
        par2 = _parity_for_state(
            rendered_prompt=p0,
            chat_format=chat_format,
            stop_tokens=new_stops,
            messages=messages,
            ref=ref,
        )
        scenarios.append(
            (
                "expanded_stop_set",
                {
                    "pv": pv2,
                    "parity": par2 if par2 is not None else parity_baseline,
                    "model": LLMCounterfactualScenario(
                        modified_system_prompt="(unchanged)",
                        modified_stop_tokens=new_stops[:80],
                        modified_chat_format=chat_format,
                        modified_assistant_anchor="(unchanged)",
                    ),
                },
            )
        )

    # 3) System tightening suffix (structural string only)
    suf = _system_tightening_suffix()
    p3 = p0 + suf
    pv3 = validate_chat_inference(
        rendered_prompt=p3,
        messages=messages,
        chat_format=chat_format,
        merged_stop_tokens=list(merged_stop_tokens),
        eos_token_str=eos_token_str,
        model_metadata=model_metadata,
        reconstruction_ok=True,
    )
    par3 = _parity_for_state(
        rendered_prompt=p3,
        chat_format=chat_format,
        stop_tokens=list(merged_stop_tokens),
        messages=messages,
        ref=ref,
    )
    scenarios.append(
        (
            "system_instruction_tightening",
            {
                "pv": pv3,
                "parity": par3 if par3 is not None else parity_baseline,
                "model": LLMCounterfactualScenario(
                    modified_system_prompt=suf.strip()[:220],
                    modified_stop_tokens=list(merged_stop_tokens),
                    modified_chat_format=chat_format,
                    modified_assistant_anchor="(unchanged)",
                ),
            },
        )
    )

    # 4) Align chat_format to LM Studio reference (single-variable) when different
    if ref:
        rfmt = str(ref.get("chat_format") or "").strip()
        if rfmt and rfmt.lower() != (chat_format or "").strip().lower():
            pv4 = validate_chat_inference(
                rendered_prompt=p0,
                messages=messages,
                chat_format=rfmt,
                merged_stop_tokens=list(merged_stop_tokens),
                eos_token_str=eos_token_str,
                model_metadata=model_metadata,
                reconstruction_ok=True,
            )
            par4 = _parity_for_state(
                rendered_prompt=p0,
                chat_format=rfmt,
                stop_tokens=list(merged_stop_tokens),
                messages=messages,
                ref=ref,
            )
            scenarios.append(
                (
                    "aligned_reference_chat_format",
                    {
                        "pv": pv4,
                        "parity": par4 if par4 is not None else parity_baseline,
                        "model": LLMCounterfactualScenario(
                            modified_system_prompt="(unchanged)",
                            modified_stop_tokens=list(merged_stop_tokens),
                            modified_chat_format=rfmt,
                            modified_assistant_anchor="(unchanged)",
                        ),
                    },
                )
            )

    for scenario_id, pack in scenarios:
        pv_s: PromptValidationResult = pack["pv"]
        parity_s: Optional[ParityReport] = pack["parity"]
        model: LLMCounterfactualScenario = pack["model"]
        new_risk = _structural_risk(pv_s)
        reduction = _clamp01(
            (base_risk - new_risk) / max(base_risk, 1e-6)
        )
        cause_new = _infer_first_token_cause_label(
            pv_s, ft_class, live_vs_gt, parity_s
        )
        p_inf, t_inf, stop_inf, _ = _compute_influence_scores(
            pv_s, parity_s, live_vs_gt
        )

        conf = _confidence_from_reduction(reduction, base_risk, new_risk)
        shift = _expected_shift_text(
            scenario_id, ft_class, cause0, cause_new
        )

        payload: dict[str, Any] = {
            "event": "llm_counterfactual_simulation",
            "scenario": scenario_id,
            "predicted_risk_reduction": round(reduction, 4),
            "expected_first_token_shift": shift,
            "confidence": conf,
            "scenario_model": asdict(model),
            "recomputed": {
                "first_token_cause_label": cause_new,
                "baseline_first_token_cause_label": cause0,
                "template_influence_score": round(t_inf, 4),
                "stop_token_influence_score": round(stop_inf, 4),
                "prompt_influence_score": round(p_inf, 4),
            },
            "baseline_structural_risk": base_risk,
            "simulated_structural_risk": new_risk,
            "observed_first_token_classification": ft_class,
            "note": "analytical_only_no_inference",
        }
        try:
            logger.info(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.debug("[counterfactual] emit failed: %s", e)
