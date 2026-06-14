"""
Instrumented sidecar titling generation for A/B inference experiments.

Generation path differs by profile; post-processing uses the existing title pipeline.
Evaluation-only diagnostics (termination, think-trace) are gated by evaluation mode.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.cognition_prompt_adapter import (
    apply_qwen3_no_think_to_prompt,
    build_cognition_prompt,
    cognition_stop_tokens,
    is_qwen3_cognition_model,
)
from core.qwen3_sidecar_inference import (
    CompletionDiagnostics,
    chat_completion_complete,
    raw_prompt_complete,
)
from core.sidecar_prompts import (
    build_title_task_parts,
    instrument_title_parse,
    task_inference_params,
)
from core.sidecar_types import SidecarTask
from core.title_inference_profiles import (
    PROFILE_IDS,
    TitleContextMode,
    TitleInferenceProfile,
    get_title_profile,
    normalize_title_context_mode,
)
from core.title_think_trace import ThinkTraceAnalysis, analyze_think_trace

logger = logging.getLogger("Qube.TitleExperiment")

_MODEL_SELECTION_PATHS = frozenset(
    {"model_line", "model_proper_phrase", "model_coerced", "post_think_tail"}
)


def is_title_evaluation_mode() -> bool:
    """When true, emit extended termination / think-trace diagnostics."""
    raw = os.environ.get("QUBE_TITLE_EVAL_MODE", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Offline evaluate tool sets this; production defaults to off.
    return bool(os.environ.get("QUBE_TITLE_EVAL_TOOL"))


@dataclass
class TitleExperimentRun:
    session_id: str = ""
    profile_id: str = ""
    profile_label: str = ""
    context_mode: TitleContextMode = "full"
    user_prompt: str = ""
    assistant_reply: str = ""
    raw_model_output: str = ""
    cleaned_model_output: str = ""
    output_char_length: int = 0
    inference_ms: float = 0.0
    had_think_block: bool = False
    think_block_stripped: bool = False
    candidates: list[dict[str, str]] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)
    final_title: str = ""
    used_fallback_repair: bool = False
    model_output_rejected: bool = False
    generation_error: str = ""
    # Evaluation-only termination diagnostics
    stop_sequences: list[str] = field(default_factory=list)
    finish_reason: str = ""
    completion_tokens: int = 0
    prompt_tokens: int = 0
    eos_encountered: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Evaluation-only think trace
    think_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _experiment_log_path() -> Path:
    base = Path(os.path.expanduser("~/.qube/logs"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "title_experiment.jsonl"


def log_title_experiment_run(run: TitleExperimentRun) -> None:
    """Structured log for every title generation (logger + optional JSONL)."""
    payload = run.to_dict()
    logger.info(
        "[TitleExperiment] session=%s profile=%s context=%s final=%r raw=%r "
        "path=%s source=%s fallback=%s rejected=%s think=%s finish=%s tokens=%s inference_ms=%.1f",
        run.session_id or "-",
        run.profile_id,
        run.context_mode,
        run.final_title,
        (run.raw_model_output or "").replace("\n", " ")[:120],
        run.selection.get("path") or "",
        run.selection.get("winner_source") or "",
        run.used_fallback_repair,
        run.model_output_rejected,
        run.think_block_stripped,
        run.finish_reason or "-",
        run.completion_tokens,
        run.inference_ms,
    )
    if is_title_evaluation_mode() or os.environ.get("QUBE_TITLE_EXPERIMENT_LOG", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        try:
            with _experiment_log_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.debug("[TitleExperiment] JSONL append failed: %s", exc)


def _apply_completion_diag(run_kwargs: dict[str, Any], diag: CompletionDiagnostics) -> None:
    run_kwargs["stop_sequences"] = list(diag.stop_sequences)
    run_kwargs["finish_reason"] = diag.finish_reason
    run_kwargs["completion_tokens"] = diag.completion_tokens
    run_kwargs["prompt_tokens"] = diag.prompt_tokens
    run_kwargs["eos_encountered"] = diag.eos_encountered
    run_kwargs["chat_template_kwargs"] = dict(diag.chat_template_kwargs)
    if diag.generation_error and not run_kwargs.get("generation_error"):
        run_kwargs["generation_error"] = diag.generation_error


def _complete_title_raw(
    model: Any,
    *,
    profile: TitleInferenceProfile,
    system: str,
    user: str,
    model_path: str,
    chat_format: str,
    max_tokens: int,
) -> tuple[str, float, CompletionDiagnostics]:
    prompt = build_cognition_prompt(
        system,
        user,
        chat_format,
        model_path=model_path if profile.use_no_think_directive else "",
    )
    if profile.use_no_think_directive and is_qwen3_cognition_model(model_path):
        prompt = apply_qwen3_no_think_to_prompt(prompt, model_path)
    stops = cognition_stop_tokens(chat_format)
    t0 = time.perf_counter()
    raw, diag = raw_prompt_complete(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=profile.temperature,
        stop=stops,
        sampling_extra={
            k: v
            for k, v in profile.sampling_kwargs(max_tokens=max_tokens).items()
            if k not in {"max_tokens", "temperature", "chat_template_kwargs"}
        },
    )
    diag.path = "raw"
    diag.stop_sequences = stops
    return raw, (time.perf_counter() - t0) * 1000.0, diag


def _complete_title_chat(
    model: Any,
    *,
    profile: TitleInferenceProfile,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, float, CompletionDiagnostics]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    template_kwargs = (
        {"enable_thinking": False}
        if profile.use_enable_thinking_false
        else {}
    )
    t0 = time.perf_counter()
    raw, diag = chat_completion_complete(
        model,
        messages,
        max_tokens=max_tokens,
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        min_p=profile.min_p,
        chat_template_kwargs=template_kwargs,
    )
    return raw, (time.perf_counter() - t0) * 1000.0, diag


def generate_title_raw(
    model: Any,
    *,
    profile: TitleInferenceProfile,
    user_prompt: str,
    assistant_reply: str = "",
    context_mode: TitleContextMode = "full",
    model_path: str = "",
    chat_format: str = "chatml",
) -> tuple[str, float, CompletionDiagnostics]:
    """Run only the generation step for a profile (for offline evaluation)."""
    system, user = build_title_task_parts(
        user_prompt,
        assistant_reply,
        context_mode=context_mode,
    )
    title_params = task_inference_params(SidecarTask.title)
    max_tokens = int(title_params.get("max_tokens", 128))
    if profile.path == "chat":
        raw, ms, diag = _complete_title_chat(
            model,
            profile=profile,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
        return raw, ms, diag
    raw, ms, diag = _complete_title_raw(
        model,
        profile=profile,
        system=system,
        user=user,
        model_path=model_path,
        chat_format=chat_format,
        max_tokens=max_tokens,
    )
    return raw, ms, diag


def run_title_generation(
    model: Any,
    *,
    profile: TitleInferenceProfile | str,
    user_prompt: str,
    assistant_reply: str = "",
    context_mode: TitleContextMode | str = "full",
    model_path: str = "",
    chat_format: str = "chatml",
    session_id: str = "",
    evaluation_mode: bool | None = None,
) -> TitleExperimentRun:
    """Full instrumented title run: generate then existing parse/fallback pipeline."""
    prof = profile if isinstance(profile, TitleInferenceProfile) else get_title_profile(profile)
    ctx = normalize_title_context_mode(context_mode)
    eval_mode = is_title_evaluation_mode() if evaluation_mode is None else evaluation_mode

    raw, inference_ms, diag = generate_title_raw(
        model,
        profile=prof,
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
        context_mode=ctx,
        model_path=model_path,
        chat_format=chat_format,
    )

    parsed = instrument_title_parse(
        raw,
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
    )
    selection_path = parsed.selection.get("path") or ""
    winner_source = parsed.selection.get("winner_source") or ""
    used_fallback = selection_path == "fallback_tournament" or (
        bool(raw.strip()) and winner_source not in _MODEL_SELECTION_PATHS
    )
    model_rejected = bool(raw.strip()) and selection_path not in _MODEL_SELECTION_PATHS

    think_trace: ThinkTraceAnalysis | None = None
    if eval_mode:
        think_trace = analyze_think_trace(
            raw,
            user_prompt=user_prompt,
            assistant_reply=assistant_reply,
            final_title=parsed.final_title,
        )

    run = TitleExperimentRun(
        session_id=session_id,
        profile_id=prof.profile_id,
        profile_label=prof.label,
        context_mode=ctx,
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
        raw_model_output=raw,
        cleaned_model_output=parsed.cleaned_output,
        output_char_length=len(raw or ""),
        inference_ms=inference_ms,
        had_think_block=parsed.had_think_block,
        think_block_stripped=parsed.think_block_stripped,
        candidates=parsed.candidates,
        selection=parsed.selection,
        final_title=parsed.final_title,
        used_fallback_repair=used_fallback,
        model_output_rejected=model_rejected,
        generation_error=diag.generation_error,
    )
    if eval_mode:
        _apply_completion_diag(run.__dict__, diag)
        if think_trace is not None:
            run.think_trace = think_trace.to_dict()
    return run


def aggregate_title_experiment_metrics(
    runs: list[TitleExperimentRun],
) -> dict[str, Any]:
    """Summary statistics grouped by profile_id."""
    by_profile: dict[str, list[TitleExperimentRun]] = {pid: [] for pid in PROFILE_IDS}
    for run in runs:
        by_profile.setdefault(run.profile_id, []).append(run)

    summary: dict[str, Any] = {"profiles": {}, "total_runs": len(runs)}
    for pid, group in by_profile.items():
        if not group:
            continue
        n = len(group)
        finish_counts: dict[str, int] = {}
        for run in group:
            key = run.finish_reason or "unknown"
            finish_counts[key] = finish_counts.get(key, 0) + 1
        summary["profiles"][pid] = {
            "count": n,
            "avg_inference_ms": sum(r.inference_ms for r in group) / n,
            "avg_output_char_length": sum(r.output_char_length for r in group) / n,
            "avg_completion_tokens": sum(r.completion_tokens for r in group) / n,
            "pct_fallback_repair": 100.0 * sum(1 for r in group if r.used_fallback_repair) / n,
            "pct_model_rejected": 100.0 * sum(1 for r in group if r.model_output_rejected) / n,
            "pct_think_block_stripped": 100.0
            * sum(1 for r in group if r.think_block_stripped)
            / n,
            "pct_empty_generation": 100.0
            * sum(1 for r in group if not (r.raw_model_output or "").strip())
            / n,
            "pct_generation_errors": 100.0
            * sum(1 for r in group if r.generation_error)
            / n,
            "finish_reason_counts": finish_counts,
        }
    return summary


def build_stop_token_analysis(runs: list[TitleExperimentRun]) -> dict[str, Any]:
    """Task 6 report: why profile A terminates early, stop/EOS behavior."""
    profile_a = [r for r in runs if r.profile_id == "A"]
    analysis: dict[str, Any] = {
        "profile_a_runs": len(profile_a),
        "profile_a_avg_raw_length": (
            sum(r.output_char_length for r in profile_a) / len(profile_a)
            if profile_a
            else 0.0
        ),
        "profile_a_finish_reasons": {},
        "profile_a_stop_sequences": cognition_stop_tokens("chatml"),
        "notes": [],
    }
    for run in profile_a:
        fr = run.finish_reason or "unknown"
        analysis["profile_a_finish_reasons"][fr] = (
            analysis["profile_a_finish_reasons"].get(fr, 0) + 1
        )
    if profile_a:
        avg_tok = sum(r.completion_tokens for r in profile_a) / len(profile_a)
        analysis["profile_a_avg_completion_tokens"] = avg_tok
        if avg_tok <= 3:
            analysis["notes"].append(
                "Profile A emits very few tokens before termination — likely stop/EOS after think opener."
            )
        if all(r.think_block_stripped for r in profile_a):
            analysis["notes"].append(
                "Profile A output is think-only; answer channel empty after strip."
            )
    analysis["notes"].append(
        "Raw path uses manual ChatML + /no_think; chat path uses GGUF Jinja via chat_handler kwargs."
    )
    analysis["notes"].append(
        f"Configured stop sequences for raw ChatML: {analysis['profile_a_stop_sequences']}"
    )
    return analysis


__all__ = [
    "TitleExperimentRun",
    "aggregate_title_experiment_metrics",
    "build_stop_token_analysis",
    "generate_title_raw",
    "is_title_evaluation_mode",
    "log_title_experiment_run",
    "run_title_generation",
]
