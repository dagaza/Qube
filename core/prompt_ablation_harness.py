"""
Diagnostic ablation harness: controlled prompt/stop variants to isolate leakage sources.

NOT production code — orchestration only for experiments (same inputs → reproducible runs).
Does not modify NativeLlamaEngine, sampling defaults in the app, or tokenizer registration.
"""
from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.execution_policy import ExecutionPolicy
from core.native_llama_inference import native_chat_completion_kwargs
from core.native_llm_debug import merge_stop_lists, reconstruct_formatted_prompt
from core.model_override_store import LearnedOverride, store_override
from core.prompt_template_router import (
    RenderPromptBundle,
    _llama_display_name,
    build_prompt_bundle,
    infer_template_type,
    resolve_reasoning_mode,
)

logger = logging.getLogger("Qube.PromptAblation")

# End-of-prompt suffixes from core/prompt_template_router.apply_reasoning_injection (for strip scenarios).
_OVERLAY_BLOCKS = (
    "\nYou may use <redacted_thinking>...</redacted_thinking> internally. Only output final answer.",
    "\nUse hidden reasoning if needed. Only output final answer.",
    "\n(Use internal reasoning. Do not expose it.)",
    "\nOnly output final answer.",
    "\nRespond with final answer only.",
    "\n\nDo not output reasoning or internal thoughts. Output final answer only.",
    "\n\nYou may use <redacted_thinking>...</redacted_thinking> for reasoning. "
    "Only final answer outside.",
)


@dataclass
class AblationScenario:
    name: str
    disable_system_suffix: bool
    disable_stop_tokens: bool
    template_override: Optional[str]
    reasoning_mode_override: Optional[str]
    strip_thinking_guidance: bool


# Canonical experiments (A–E).
SCENARIO_BASELINE = AblationScenario(
    "BASELINE",
    disable_system_suffix=False,
    disable_stop_tokens=False,
    template_override=None,
    reasoning_mode_override=None,
    strip_thinking_guidance=False,
)
SCENARIO_NO_SYSTEM = AblationScenario(
    "NO_SYSTEM_CONSTRAINT",
    disable_system_suffix=True,
    disable_stop_tokens=False,
    template_override=None,
    reasoning_mode_override=None,
    strip_thinking_guidance=True,
)
SCENARIO_NO_STOPS = AblationScenario(
    "NO_STOP_TOKENS",
    disable_system_suffix=False,
    disable_stop_tokens=True,
    template_override=None,
    reasoning_mode_override=None,
    strip_thinking_guidance=False,
)
SCENARIO_TEMPLATE_PURE = AblationScenario(
    "TEMPLATE_PURE",
    disable_system_suffix=False,
    disable_stop_tokens=True,
    template_override=None,
    reasoning_mode_override="hard",
    strip_thinking_guidance=False,
)
SCENARIO_THINKING_INJECT = AblationScenario(
    "THINKING_INJECTION_TEST",
    disable_system_suffix=False,
    disable_stop_tokens=False,
    template_override=None,
    reasoning_mode_override="soft",
    strip_thinking_guidance=False,
)

DEFAULT_SCENARIOS: tuple[AblationScenario, ...] = (
    SCENARIO_BASELINE,
    SCENARIO_NO_SYSTEM,
    SCENARIO_NO_STOPS,
    SCENARIO_TEMPLATE_PURE,
    SCENARIO_THINKING_INJECT,
)

# Soft policy for injection test (always allow thinking guidance path).
_POLICY_FORCED_SOFT = ExecutionPolicy(
    execution_mode="thinking",
    allow_thinking_tokens=True,
    strip_thinking_output=False,
    ui_display_thinking=True,
    tts_strip_thinking=False,
    enforcement_mode="soft",
)


@dataclass
class ScenarioRunResult:
    scenario: str
    first_token: str
    first_20_tokens: str
    full_text_sample: str
    has_redacted_thinking: bool
    has_we_need: bool
    has_lets: bool
    meta_commentary_detected: bool
    leakage_detected: bool
    execution_policy_snapshot: Dict[str, Any]
    stop_token_count: int
    prompt_char_len: int


@dataclass
class AblationReport:
    scenario_results: List[ScenarioRunResult] = field(default_factory=list)
    divergence_matrix: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    first_token_drift_map: Dict[str, str] = field(default_factory=dict)
    template_sensitivity_score: float = 0.0
    stop_token_dependency_score: float = 0.0
    reasoning_leak_trigger_source: str = "insufficient_evidence"


def infer_override_from_ablation(
    report: AblationReport, model_name: str
) -> Optional[LearnedOverride]:
    baseline = next((r for r in report.scenario_results if r.scenario == "BASELINE"), None)
    if not baseline:
        return None

    # first_token is a single character from streaming; phrases need full/head text.
    sample_lc = (baseline.full_text_sample or "").lower()
    head_lc = ((baseline.first_20_tokens or "") + (baseline.first_token or "")).lower()
    probe_lc = sample_lc if sample_lc else head_lc
    first = baseline.first_token.lower()

    extra_stops: list[str] = []
    enforce_anchor = False
    strip_thinking: Optional[bool] = None
    enforcement_mode: Optional[str] = None
    force_execution_mode: Optional[str] = None

    # CASE 1: Forced thinking leakage
    if (
        "<redacted_thinking>" in probe_lc
        or "let's" in probe_lc
        or "we need to" in probe_lc
    ):
        extra_stops += [
            "<redacted_thinking>",
            "</redacted_thinking>",
            "<thinking>",
            "</thinking>",
        ]
        strip_thinking = True
        enforcement_mode = "hard"
        force_execution_mode = "direct"

    # CASE 2: Missing assistant anchor
    if baseline.first_token.strip() == "" or len(first) == 0:
        enforce_anchor = True

    # CASE 3: Meta commentary leakage
    if baseline.meta_commentary_detected:
        strip_thinking = True
        enforcement_mode = "hard"

    # CASE 4: Stop token dependency failure (first token unchanged without stops)
    if not report.stop_token_dependency_score:
        extra_stops += ["</s>", "<|end|>"]

    if not any(
        [
            extra_stops,
            enforce_anchor,
            strip_thinking,
            enforcement_mode,
            force_execution_mode,
        ]
    ):
        return None

    return LearnedOverride(
        model_name=model_name,
        force_execution_mode=force_execution_mode,
        enforcement_mode=enforcement_mode,
        strip_thinking=strip_thinking,
        extra_stop_tokens=list(dict.fromkeys(extra_stops)),
        enforce_assistant_anchor=enforce_anchor,
    )


def _strip_overlay_suffixes(prompt: str) -> str:
    p = prompt
    changed = True
    while changed:
        changed = False
        for block in _OVERLAY_BLOCKS:
            if p.endswith(block):
                p = p[: -len(block)]
                changed = True
    return p.rstrip()


def _formatter_only_stops(fmt_stop: Any) -> List[str]:
    merged, _ = merge_stop_lists(None, fmt_stop)
    return list(merged)


def _build_bundle_for_scenario(
    scenario: AblationScenario,
    llama: Any,
    messages: list[dict],
    model_profile: Any,
    execution_policy: ExecutionPolicy,
    baseline: RenderPromptBundle,
    raw_prompt: Optional[str],
    fmt_stop: Any,
) -> RenderPromptBundle:
    """Produce a RenderPromptBundle for this scenario without mutating llama chat_format (template_override reserved)."""
    cf = str(getattr(llama, "chat_format", "") or "")
    tt = infer_template_type(llama)

    if scenario.name == "THINKING_INJECTION_TEST":
        b, _, _ = build_prompt_bundle(llama, messages, model_profile, _POLICY_FORCED_SOFT)
        return b

    if scenario.name == "TEMPLATE_PURE":
        rp = raw_prompt or ""
        stops = _formatter_only_stops(fmt_stop)
        return RenderPromptBundle(
            prompt=rp,
            chat_format=cf,
            stop_tokens=stops,
            template_type=tt,
            reasoning_mode="hard",
        )

    # Baseline-derived
    prompt = baseline.prompt
    stops = list(baseline.stop_tokens)

    if scenario.disable_system_suffix or scenario.strip_thinking_guidance:
        prompt = _strip_overlay_suffixes(prompt)

    if scenario.disable_stop_tokens:
        stops = _formatter_only_stops(fmt_stop)

    rm = baseline.reasoning_mode
    if scenario.reasoning_mode_override:
        rm = scenario.reasoning_mode_override

    return RenderPromptBundle(
        prompt=prompt,
        chat_format=cf,
        stop_tokens=stops,
        template_type=tt,
        reasoning_mode=rm,
    )


_META_PATTERNS = (
    re.compile(r"(?is)provide\s+(?:a|an|the)?\s*(?:concise\s+)?(?:answer|response|summary)\b"),
    re.compile(r"(?is)\bwe\s+should\b"),
    re.compile(r"(?is)\bstep\s*1\b"),
    re.compile(r"(?is)\bhere(?:'s| is)\s+(?:my|a|an)\s+(?:answer|response)\b"),
)


def _analyze_leakage(text: str) -> tuple[bool, bool, bool, bool, bool]:
    t = text or ""
    low = t.lower()
    has_rt = "<redacted_thinking>" in t or "</redacted_thinking>" in t
    has_we = "we need to" in low
    has_lets = "let's" in low or "let’s" in low
    meta = any(p.search(t) for p in _META_PATTERNS)
    leakage = has_rt or has_we or has_lets or meta
    return has_rt, has_we, has_lets, meta, leakage


def _stream_completion_text(
    llama: Any,
    prompt: str,
    stop_tokens: List[str],
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    """Accumulate streaming create_completion (prompt-string path; isolated from chat handler)."""
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
        "echo": False,
    }
    if stop_tokens:
        kwargs["stop"] = stop_tokens
    kw = dict(kwargs)
    kw["seed"] = int(seed)
    try:
        stream = llama.create_completion(**kw)
    except TypeError:
        stream = llama.create_completion(**kwargs)
    parts: List[str] = []
    for chunk in stream:
        ch = chunk.get("choices", [{}])[0]
        txt = ch.get("text") or ""
        if txt:
            parts.append(txt)
    return "".join(parts)


def _first_token_and_20(full: str) -> tuple[str, str]:
    if not full:
        return "", ""
    tok = full[0]
    return tok, full[:20]


def run_ablation_test(
    llama: Any,
    messages: list[dict],
    model_profile: Any,
    execution_policy: ExecutionPolicy,
    *,
    scenarios: tuple[AblationScenario, ...] = DEFAULT_SCENARIOS,
    max_tokens: int = 96,
    temperature: float = 0.7,
    seed: int = 42,
    model_name: Optional[str] = None,
) -> AblationReport:
    """
    Run each scenario once via ``llama.create_completion`` on the constructed prompt string.

    Deterministic given same model weights, seed (if backend honors it), and inputs.
    """
    raw_prompt, fmt_stop, _ = reconstruct_formatted_prompt(llama, messages)
    baseline, _, _ = build_prompt_bundle(llama, messages, model_profile, execution_policy)

    results: List[ScenarioRunResult] = []
    policy_snap = {
        "execution_mode": execution_policy.execution_mode,
        "allow_thinking_tokens": execution_policy.allow_thinking_tokens,
        "enforcement_mode": execution_policy.enforcement_mode,
        "reasoning_mode_resolved": resolve_reasoning_mode(execution_policy),
    }

    for sc in scenarios:
        bundle = _build_bundle_for_scenario(
            sc,
            llama,
            messages,
            model_profile,
            execution_policy,
            baseline,
            raw_prompt,
            fmt_stop,
        )
        text = _stream_completion_text(
            llama,
            bundle.prompt,
            bundle.stop_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        ft, f20 = _first_token_and_20(text)
        h_rt, h_we, h_lets, h_meta, leakage = _analyze_leakage(text)

        logger.info(
            "[LLM-ABLATION] scenario=%s first_token=%r leakage=%s",
            sc.name,
            ft,
            leakage,
        )

        snap = dict(policy_snap)
        snap["scenario"] = sc.name
        snap["bundle_reasoning_mode"] = bundle.reasoning_mode
        snap["template_type"] = bundle.template_type

        results.append(
            ScenarioRunResult(
                scenario=sc.name,
                first_token=ft,
                first_20_tokens=f20,
                full_text_sample=text[:2000],
                has_redacted_thinking=h_rt,
                has_we_need=h_we,
                has_lets=h_lets,
                meta_commentary_detected=h_meta,
                leakage_detected=leakage,
                execution_policy_snapshot=snap,
                stop_token_count=len(bundle.stop_tokens),
                prompt_char_len=len(bundle.prompt or ""),
            )
        )

    # first_token drift
    drift = {r.scenario: r.first_token for r in results}
    names = [r.scenario for r in results]
    div: Dict[str, Dict[str, bool]] = {a: {} for a in names}
    for a, b in itertools.product(names, repeat=2):
        ta = drift.get(a, "")
        tb = drift.get(b, "")
        div[a][b] = ta != tb

    uniq_first = len({drift[n] for n in names if drift.get(n) is not None})
    n = max(len(names), 1)
    template_score = 0.0 if uniq_first <= 1 else min(1.0, (uniq_first - 1) / max(n - 1, 1))

    baseline_ft = drift.get("BASELINE", "")
    no_stops_ft = drift.get("NO_STOP_TOKENS", "")
    stop_dep = 1.0 if baseline_ft != no_stops_ft else 0.0

    # Heuristic best guess for leak trigger
    base_leak = next((r.leakage_detected for r in results if r.scenario == "BASELINE"), False)
    pure_leak = next((r.leakage_detected for r in results if r.scenario == "TEMPLATE_PURE"), False)
    no_sys_leak = next((r.leakage_detected for r in results if r.scenario == "NO_SYSTEM_CONSTRAINT"), False)
    guess = "insufficient_evidence"
    if base_leak and not pure_leak:
        guess = "likely_template_or_policy_overlay (baseline leaks, template pure does not)"
    elif base_leak and pure_leak:
        guess = "likely_model_or_base_template (leakage survives template-pure prompt)"
    elif base_leak and not no_sys_leak:
        guess = "likely_system_suffix_or_guidance_strings"
    elif not base_leak:
        guess = "no_clear_leakage_in_heuristic_probe"

    report = AblationReport(
        scenario_results=results,
        divergence_matrix=div,
        first_token_drift_map=drift,
        template_sensitivity_score=template_score,
        stop_token_dependency_score=stop_dep,
        reasoning_leak_trigger_source=guess,
    )

    mn = (model_name or "").strip()
    if not mn and model_profile is not None:
        mn = str(getattr(model_profile, "model_name", "") or "").strip()
    if not mn:
        mn = _llama_display_name(llama)

    override = infer_override_from_ablation(report, mn)
    if override:
        store_override(override)
        logger.info(
            "[LLM-SELF-HEAL] learned_override model=%s stops=%d anchor=%s enforcement=%s",
            mn,
            len(override.extra_stop_tokens),
            override.enforce_assistant_anchor,
            override.enforcement_mode,
        )

    return report
