"""
Post-inference execution causality mapping for native LLM debugging.

Correlates prompt structure, chat template, stops, sampler ground truth, and live traces
into a single attribution report. Observer-only: no prompt/sampling changes.

Enable with QUBE_LLM_CAUSALITY=1.

Optional LM Studio reference JSON (QUBE_LLM_REFERENCE_JSON) may include:
  first_sampler_token_ids: list[int]   # first 1–3 tokens from a Studio run (same model)
  first_token_texts: list[str]         # optional parallel decoded pieces
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

from core.native_sampler_gt import (
    detokenize_sampler_token_ids,
    sequence_prefix_match_score,
)
from core.native_token_trace import classify_first_token_piece, extract_first_n_generation_tokens
from core.prompt_integrity_validator import ParityReport, PromptValidationResult

logger = logging.getLogger("Qube.NativeLLM.Debug")


def causality_report_enabled() -> bool:
    return os.environ.get("QUBE_LLM_CAUSALITY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class LLMExecutionCausalReport:
    prompt_influence_score: float
    template_influence_score: float
    stop_token_influence_score: float
    sampler_stability_score: float
    first_token_cause_label: str
    divergence_root_cause: str


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _first_token_classification(
    llama: Any,
    assistant_text: str,
    ground_truth_token_ids: Optional[list[int]],
    prompt_tokens: Optional[list[int]],
) -> tuple[str, list[int], str]:
    """
    Returns (classification_label, first_n_ids_used, first_piece_str).
    Prefers sampler GT ids; falls back to post-hoc tokenize of assistant prefix.
    """
    n = 3
    if (
        llama is not None
        and ground_truth_token_ids
        and prompt_tokens is not None
    ):
        ids = ground_truth_token_ids[:n]
        pieces = detokenize_sampler_token_ids(llama, prompt_tokens, ids[:1])
        piece = pieces[0] if pieces else ""
        cls = classify_first_token_piece(piece)
        return cls, ids[:1], piece

    if llama is not None and assistant_text:
        ids, pieces, _ = extract_first_n_generation_tokens(llama, assistant_text, 1)
        piece = pieces[0] if pieces else ""
        cls = classify_first_token_piece(piece)
        return cls, ids[:1], piece

    return "unknown", [], ""


def _infer_first_token_cause_label(
    pv: PromptValidationResult,
    ft_class: str,
    live_vs_gt: Optional[float],
    parity: Optional[ParityReport],
) -> str:
    if not pv.assistant_anchor_present:
        return "missing_assistant_anchor_strength"
    if ft_class == "role_continuation":
        return "template_boundary_shift"
    if ft_class == "instruction_echo":
        return "instruction_layout_echo"
    if not pv.eos_token_present or pv.stop_tokens_suspicious:
        return "stop_configuration_gap"
    if live_vs_gt is not None and live_vs_gt < 0.999:
        return "stream_reconstruction_drift"
    if parity is not None and parity.score < 0.85:
        return "lm_studio_structural_divergence"
    if ft_class in ("reasoning_prefix", "unknown"):
        return "ambiguous_opener_heuristic"
    return "expected_generation"


def _compute_influence_scores(
    pv: PromptValidationResult,
    parity: Optional[ParityReport],
    live_vs_gt: Optional[float],
) -> tuple[float, float, float, float]:
    """Prompt / template / stop / sampler influence weights (0..1 heuristic)."""
    # Prompt: stronger when anchor missing or bleed-like first token
    prompt_w = _clamp01(
        (0.75 if not pv.assistant_anchor_present else 0.25)
        + (0.2 if "missing_assistant_generation_anchor" in pv.risk_flags else 0.0)
        + (0.15 if "possible_prompt_bleed_multi_turn" in pv.risk_flags else 0.0)
    )

    # Template: confidence inverse + parity format drift
    tmpl_penalty = 1.0 - float(pv.chat_template_confidence or 0.0)
    fmt_drift = 0.0
    if parity and parity.differences:
        if any("chat_format" in d for d in parity.differences):
            fmt_drift = 0.35
    template_w = _clamp01(0.4 * tmpl_penalty + fmt_drift)

    # Stops / EOS
    stop_w = _clamp01(
        (0.45 if not pv.eos_token_present else 0.0)
        + (0.35 if pv.stop_tokens_suspicious else 0.0)
        + (0.15 if "eos_token_not_in_merged_stops_list" in pv.risk_flags else 0.0)
    )

    # Sampler path stability (live re-tokenize vs GT); unknown middle if missing
    if live_vs_gt is None:
        sampler_w = 0.5
    else:
        sampler_w = _clamp01(float(live_vs_gt))

    return prompt_w, template_w, stop_w, sampler_w


def _build_divergence_root_cause(
    pv: PromptValidationResult,
    parity: Optional[ParityReport],
    cause_label: str,
) -> str:
    parts: list[str] = []
    if parity and parity.differences:
        parts.append(parity.likely_root_cause)
    if not pv.assistant_anchor_present:
        parts.append("Assistant generation slot is weak or missing in the rendered prompt.")
    if pv.stop_tokens_suspicious or not pv.eos_token_present:
        parts.append("Stop/EOS configuration may not match the server or formatter.")
    if cause_label == "stream_reconstruction_drift":
        parts.append(
            "Live delta tokenization diverged from sampler ids — diagnose post-stream, not sampling."
        )
    if not parts:
        return "No dominant causal factor flagged; output aligns with structural checks."
    return " ".join(parts)


def _confidence(
    pv: PromptValidationResult,
    parity: Optional[ParityReport],
    live_vs_gt: Optional[float],
    ref_first_match: Optional[float],
) -> float:
    xs: list[float] = [float(pv.chat_template_confidence or 0.0)]
    if parity is not None:
        xs.append(float(parity.score))
    if live_vs_gt is not None:
        xs.append(float(live_vs_gt))
    if ref_first_match is not None:
        xs.append(float(ref_first_match))
    if not xs:
        return 0.5
    return round(sum(xs) / len(xs), 4)


def _key_factors(
    pv: PromptValidationResult,
    parity: Optional[ParityReport],
    cause_label: str,
) -> list[str]:
    out: list[str] = []
    if parity and parity.differences:
        if any("chat_format" in d for d in parity.differences):
            out.append("chat_format divergence")
        if any("stop_tokens" in d for d in parity.differences):
            out.append("stop token mismatch")
        if any("rendered_prompt" in d for d in parity.differences):
            out.append("rendered prompt mismatch")
    if not pv.assistant_anchor_present:
        out.append("missing assistant anchor")
    if pv.stop_tokens_suspicious:
        out.append("stop set heuristic weak")
    if cause_label == "stream_reconstruction_drift":
        out.append("live vs sampler token id drift")
    return list(dict.fromkeys(out))[:8]


def _recommended_actions(pv: PromptValidationResult, cause_label: str) -> list[str]:
    acts: list[str] = []
    if not pv.assistant_anchor_present:
        acts.append(
            "Strengthen the assistant turn boundary in the chat template (explicit assistant header)."
        )
    if not pv.eos_token_present:
        acts.append("Ensure EOS string appears in merged stop list for parity with LM Studio.")
    if cause_label == "stream_reconstruction_drift":
        acts.append("Trust sampler ground truth over live delta tokenization when they disagree.")
    if cause_label == "lm_studio_structural_divergence":
        acts.append("Export LM Studio rendered prompt + stops to QUBE_LLM_REFERENCE_JSON and diff.")
    if not acts:
        acts.append("No automatic fix inferred; use parity logs and GT trace for next steps.")
    return acts[:5]


def build_execution_causal_report(
    *,
    pv: PromptValidationResult,
    parity: Optional[ParityReport],
    assistant_text: str,
    llama: Any,
    ground_truth_token_ids: Optional[list[int]],
    prompt_tokens: Optional[list[int]],
    live_token_ids: Optional[list[int]],
) -> LLMExecutionCausalReport:
    """Assemble scores and labels (post-inference)."""
    live_vs_gt: Optional[float] = None
    if (
        live_token_ids is not None
        and ground_truth_token_ids is not None
        and ground_truth_token_ids
    ):
        cap = min(32, len(ground_truth_token_ids), len(live_token_ids))
        live_vs_gt = sequence_prefix_match_score(
            live_token_ids[:cap], ground_truth_token_ids[:cap]
        )

    ft_class, _, _ = _first_token_classification(
        llama, assistant_text, ground_truth_token_ids, prompt_tokens
    )
    cause = _infer_first_token_cause_label(pv, ft_class, live_vs_gt, parity)
    p, t, s, samp = _compute_influence_scores(pv, parity, live_vs_gt)
    div = _build_divergence_root_cause(pv, parity, cause)

    return LLMExecutionCausalReport(
        prompt_influence_score=p,
        template_influence_score=t,
        stop_token_influence_score=s,
        sampler_stability_score=samp,
        first_token_cause_label=cause,
        divergence_root_cause=div,
    )


def _reference_first_three_match(
    native_gt: list[int], ref: dict[str, Any]
) -> Optional[float]:
    ref_ids = ref.get("first_sampler_token_ids")
    if not isinstance(ref_ids, list) or not ref_ids:
        return None
    try:
        r = [int(x) for x in ref_ids[:3]]
    except (TypeError, ValueError):
        return None
    if not native_gt:
        return None
    n = min(3, len(r), len(native_gt))
    if n == 0:
        return None
    return sequence_prefix_match_score(native_gt[:n], r[:n])


def emit_execution_causality_report(
    report: LLMExecutionCausalReport,
    *,
    pv: PromptValidationResult,
    parity: Optional[ParityReport],
    assistant_text: str,
    llama: Any,
    ground_truth_token_ids: Optional[list[int]],
    prompt_tokens: Optional[list[int]],
    live_token_ids: Optional[list[int]],
    lm_studio_reference: Optional[dict[str, Any]],
    chat_format: str,
) -> None:
    """Emit a single JSON line: llm_execution_causality_report."""
    live_vs_gt: Optional[float] = None
    if (
        live_token_ids is not None
        and ground_truth_token_ids is not None
        and ground_truth_token_ids
    ):
        cap = min(32, len(ground_truth_token_ids), len(live_token_ids))
        live_vs_gt = sequence_prefix_match_score(
            live_token_ids[:cap], ground_truth_token_ids[:cap]
        )

    ft_class, gt_first_ids, first_piece = _first_token_classification(
        llama, assistant_text, ground_truth_token_ids, prompt_tokens
    )
    ref_match = None
    if lm_studio_reference and ground_truth_token_ids:
        ref_match = _reference_first_three_match(
            ground_truth_token_ids, lm_studio_reference
        )

    structural_impact: Optional[float] = None
    if parity is not None:
        structural_impact = round(1.0 - float(parity.score), 4)

    first_three_diff: Optional[float] = None
    if ref_match is not None:
        first_three_diff = round(1.0 - float(ref_match), 4)

    conf = _confidence(pv, parity, live_vs_gt, ref_match)

    cause_key = report.first_token_cause_label
    # Map to shorter event keys for readability
    first_token_cause = cause_key
    root_cause = report.divergence_root_cause

    payload: dict[str, Any] = {
        "event": "llm_execution_causality_report",
        "first_token_cause": first_token_cause,
        "root_cause": root_cause,
        "confidence": conf,
        "key_factors": _key_factors(pv, parity, cause_key),
        "report": asdict(report),
        "first_token_classification": ft_class,
        "first_token_piece": first_piece[:120] if first_piece else "",
        "gt_first_token_ids": gt_first_ids,
        "lm_studio_structural_diff_impact": structural_impact,
        "first_three_token_differential_impact": first_three_diff,
        "reference_first_three_match_score": ref_match,
        "live_vs_sampler_prefix_score": live_vs_gt,
        "chat_format": chat_format,
        "recommended_actions": _recommended_actions(pv, cause_key),
        "predictable_issue": bool(conf >= 0.75 and cause_key != "expected_generation"),
        "automatic_fix_available": False,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def maybe_emit_execution_causality_report(
    llama: Any,
    *,
    assistant_text: str,
    prompt_validation: PromptValidationResult,
    parity: Optional[ParityReport],
    trace_preflight: dict[str, Any],
    chat_format: str,
    lm_studio_reference: Optional[dict[str, Any]],
    live_token_ids: Optional[list[int]],
    ground_truth_token_ids: Optional[list[int]],
    prompt_tokens: Optional[list[int]],
) -> None:
    """Entry point from NativeLlamaEngine after inference (post-stream)."""
    if not causality_report_enabled():
        return
    _ = trace_preflight
    try:
        report = build_execution_causal_report(
            pv=prompt_validation,
            parity=parity,
            assistant_text=assistant_text,
            llama=llama,
            ground_truth_token_ids=ground_truth_token_ids,
            prompt_tokens=prompt_tokens,
            live_token_ids=live_token_ids,
        )
        emit_execution_causality_report(
            report,
            pv=prompt_validation,
            parity=parity,
            assistant_text=assistant_text,
            llama=llama,
            ground_truth_token_ids=ground_truth_token_ids,
            prompt_tokens=prompt_tokens,
            live_token_ids=live_token_ids,
            lm_studio_reference=lm_studio_reference,
            chat_format=chat_format,
        )
    except Exception as e:
        logger.debug("[causality] emit failed: %s", e)
