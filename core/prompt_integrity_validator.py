"""
Deterministic prompt structure checks + LM Studio parity heuristics (native path).

Does not alter prompts, sampling, or outputs — validation and logging only.

Enable JSON validation lines with QUBE_LLM_DEBUG=1 or QUBE_LLM_PROMPT_VALIDATE=1.

Optional LM Studio baseline JSON path: QUBE_LLM_REFERENCE_JSON=/path/to/ref.json
Expected keys: rendered_prompt, chat_format, stop_tokens (list[str]).
Optional (causality / first-token diff): first_sampler_token_ids (list[int], first 1–3 tokens).
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal, Optional

logger = logging.getLogger("Qube.NativeLLM.Debug")

Verdict = Literal["OK", "SUSPECT", "BROKEN"]


@dataclass
class PromptValidationResult:
    assistant_anchor_present: bool
    assistant_role_correctly_terminated: bool
    user_message_closed_properly: bool
    eos_token_present: bool
    stop_tokens_suspicious: bool
    retrieval_injection_risk: bool
    chat_template_confidence: float
    risk_flags: list[str] = field(default_factory=list)
    verdict: str = "OK"
    # Native engine: model reasoning profile (observability only; no inference change)
    model_reasoning_profile_detected: bool = False
    execution_mode: str = "unknown"


@dataclass
class ParityReport:
    score: float
    differences: list[str]
    likely_root_cause: str


def prompt_validation_log_enabled() -> bool:
    if os.environ.get("QUBE_LLM_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return os.environ.get("QUBE_LLM_PROMPT_VALIDATE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def load_lm_studio_reference_from_env() -> Optional[dict[str, Any]]:
    p = (os.environ.get("QUBE_LLM_REFERENCE_JSON") or "").strip()
    if not p:
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[prompt_validation] QUBE_LLM_REFERENCE_JSON load failed: %s", e)
        return None


def native_snapshot_for_parity(
    *,
    rendered_prompt: str,
    chat_format: str,
    stop_tokens: list[str],
    messages: list[dict],
) -> dict[str, Any]:
    sys_msgs = [m for m in messages if (m.get("role") or "").lower() == "system"]
    return {
        "rendered_prompt": rendered_prompt or "",
        "chat_format": chat_format,
        "stop_tokens": list(stop_tokens),
        "has_system_message": len(sys_msgs) > 0,
        "system_chars": sum(len(str(m.get("content", ""))) for m in sys_msgs),
    }


def validate_chat_inference(
    *,
    rendered_prompt: Optional[str],
    messages: list[dict],
    chat_format: str,
    merged_stop_tokens: list[str],
    eos_token_str: str,
    model_metadata: dict[str, Any],
    reconstruction_ok: bool,
    model_reasoning_profile_detected: bool = False,
    execution_mode: str = "unknown",
) -> PromptValidationResult:
    """
    Structural checks on the same rendered prompt string the native engine would tokenize
    (when reconstruction succeeds).
    """
    flags: list[str] = []
    p = rendered_prompt or ""
    cf = (chat_format or "").strip().lower()

    # --- confidence ---
    if reconstruction_ok and p:
        conf = 1.0 if model_metadata.get("tokenizer.chat_template") else 0.85
    elif p:
        conf = 0.55
    else:
        conf = 0.2
        flags.append("prompt_reconstruction_missing")

    # --- assistant anchor (generation slot) ---
    anchor = _has_assistant_anchor(p, cf)
    if not anchor and p:
        flags.append("missing_assistant_generation_anchor")
    if not p:
        flags.append("empty_rendered_prompt")

    # --- user / role closure (template-family heuristics) ---
    user_closed = _user_turn_closed_heuristic(p, cf)
    if not user_closed and p and _expects_chatml_style(p, cf):
        flags.append("user_turn_may_be_unclosed")

    assist_term = _assistant_prefix_well_formed(p, cf)
    if not assist_term and anchor:
        flags.append("assistant_header_format_uncertain")

    # --- EOS in stops (string-level; token-id-only EOS is invisible here) ---
    eos_ok = True
    if eos_token_str:
        eos_ok = bool(
            merged_stop_tokens
            and any(eos_token_str in s for s in merged_stop_tokens)
        )
        if not eos_ok:
            flags.append("eos_token_not_in_merged_stops_list")

    # --- suspicious stops ---
    suspicious = _stops_suspicious(merged_stop_tokens, eos_token_str, cf)
    if suspicious:
        flags.append("stop_set_may_be_incomplete")

    # --- retrieval ---
    retr = _retrieval_injection_risk(messages)
    if retr:
        flags.append("retrieval_injection_structure_risk")

    # --- bleed / meta heuristic (structural only) ---
    if not anchor and cf and cf not in ("", "llama-2"):
        flags.append("possible_prompt_bleed_multi_turn")
    if suspicious and anchor:
        flags.append("possible_meta_output_weak_stops")

    verdict: Verdict = "OK"
    if not anchor or not p:
        verdict = "BROKEN"
    elif len(flags) >= 2 or retr or (not user_closed and _expects_chatml_style(p, cf)):
        verdict = "SUSPECT"
    elif flags:
        verdict = "SUSPECT"

    return PromptValidationResult(
        assistant_anchor_present=anchor,
        assistant_role_correctly_terminated=assist_term,
        user_message_closed_properly=user_closed,
        eos_token_present=eos_ok,
        stop_tokens_suspicious=suspicious,
        retrieval_injection_risk=retr,
        chat_template_confidence=round(conf, 4),
        risk_flags=flags,
        verdict=verdict,
        model_reasoning_profile_detected=model_reasoning_profile_detected,
        execution_mode=execution_mode,
    )


def _expects_chatml_style(p: str, cf: str) -> bool:
    if "chatml" in cf or "im_start" in p:
        return True
    return False


def _has_assistant_anchor(p: str, cf: str) -> bool:
    if not p.strip():
        return False
    tail = p[-1200:] if len(p) > 1200 else p
    # ChatML / Qwen-style
    if re.search(r"<\|im_start\|\s*assistant\s*\n", tail, re.I):
        return True
    # Llama 3
    if re.search(
        r"<\|start_header_id\|>\s*assistant\s*<\|end_header_id\|>",
        tail,
        re.I | re.DOTALL,
    ):
        return True
    # Mistral instruct: trailing [/INST] opens assistant generation
    if cf == "mistral-instruct" or (cf == "llama-2" and "[INST]" in p):
        if re.search(r"\[/INST]\s*$", p.strip()):
            return True
    # Alpaca / generic: assistant slot
    if re.search(r"###\s*Response\s*\n?\s*$", p, re.I):
        return True
    # Fallback: any 'assistant' role marker near end
    if "assistant" in tail.lower() and (
        "im_start" in tail or "header_id" in tail or "[/INST]" in tail[-80:]
    ):
        return True
    return False


def _assistant_prefix_well_formed(p: str, cf: str) -> bool:
    if not p:
        return False
    if "<|im_start|>" in p and re.search(r"<\|im_start\|\s*assistant\s*\n\s*$", p[-400:], re.I):
        return True
    if "start_header_id" in p and "assistant" in p[-500:].lower():
        return True
    if "[/INST]" in p[-120:]:
        return True
    return _has_assistant_anchor(p, cf)


def _user_turn_closed_heuristic(p: str, cf: str) -> bool:
    if not _expects_chatml_style(p, cf):
        return True
    # Last user segment should be followed by im_end before assistant
    if "<|im_start|>" not in p:
        return True
    # Find last user block
    parts = re.split(r"<\|im_start\|\s*user\s*\n", p, flags=re.I)
    if len(parts) < 2:
        return True
    last_user_block = parts[-1]
    if "assistant" in last_user_block[:200].lower():
        return True
    return bool(re.search(r"<\|im_end\|>", last_user_block))


def _stops_suspicious(stops: list[str], eos: str, cf: str) -> bool:
    if not stops:
        return True
    if eos and not any(eos in s for s in stops):
        # Formatter may rely on token-id EOS only; flag for LM Studio string parity checks
        return True
    if len(stops) < 2:
        return True
    return False


def _retrieval_injection_risk(messages: list[dict]) -> bool:
    for m in messages:
        c = str(m.get("content") or "")
        if "=== SYSTEM RETRIEVED CONTEXT ===" in c:
            if "USER QUERY:" not in c and "================================" not in c:
                return True
    return False


def compute_parity_score(
    native: dict[str, Any],
    reference_lm_studio: dict[str, Any],
) -> ParityReport:
    """
    Compare native snapshot dict vs LM Studio reference (same keys as native_snapshot_for_parity).
    """
    diffs: list[str] = []
    n_prompt = str(native.get("rendered_prompt") or "")
    r_prompt = str(reference_lm_studio.get("rendered_prompt") or "")
    n_fmt = str(native.get("chat_format") or "").lower()
    r_fmt = str(reference_lm_studio.get("chat_format") or "").lower()

    score = 1.0
    if n_fmt != r_fmt:
        diffs.append(f"chat_format: native={n_fmt!r} reference={r_fmt!r}")
        score -= 0.28

    ns = native.get("stop_tokens") or []
    rs = reference_lm_studio.get("stop_tokens") or []
    if not isinstance(ns, list):
        ns = []
    if not isinstance(rs, list):
        rs = []
    ns_s, rs_s = set(map(str, ns)), set(map(str, rs))
    if ns_s or rs_s:
        inter = len(ns_s & rs_s)
        union = len(ns_s | rs_s) or 1
        jacc = inter / union
        if jacc < 0.85:
            diffs.append(
                f"stop_tokens_jaccard={jacc:.2f} native_count={len(ns)} reference_count={len(rs)}"
            )
            score -= 0.25 * (1.0 - jacc)

    ratio = SequenceMatcher(a=n_prompt, b=r_prompt).ratio()
    if ratio < 0.97:
        diffs.append(f"rendered_prompt_similarity={ratio:.3f}")
        score -= 0.35 * (1.0 - ratio)

    nh = bool(native.get("has_system_message"))
    rh = bool(reference_lm_studio.get("has_system_message"))
    if nh != rh:
        diffs.append(f"has_system_message: native={nh} reference={rh}")
        score -= 0.08

    score = max(0.0, min(1.0, score))

    cause = "No major structural differences detected."
    if diffs:
        if "chat_format" in diffs[0]:
            cause = "Chat template / chat_format string differs from LM Studio reference."
        elif "stop_tokens" in diffs[0]:
            cause = "Stop token set differs — completion boundaries may not match LM Studio."
        elif "rendered_prompt" in diffs[0]:
            cause = "Rendered prompt body differs — role markers or injection layout may diverge."
        elif "has_system_message" in diffs[0]:
            cause = "System message presence differs between native and reference."

    return ParityReport(score=round(score, 4), differences=diffs, likely_root_cause=cause)


def log_prompt_validation_jsonlines(
    result: PromptValidationResult,
    parity: Optional[ParityReport],
    *,
    chat_format: str,
    merged_stop_count: int,
    reconstruction_note: str,
) -> None:
    if not prompt_validation_log_enabled():
        return
    payload: dict[str, Any] = {
        "event": "llm_prompt_validation",
        "verdict": result.verdict,
        "assistant_anchor_present": result.assistant_anchor_present,
        "assistant_role_correctly_terminated": result.assistant_role_correctly_terminated,
        "user_message_closed_properly": result.user_message_closed_properly,
        "eos_token_present": result.eos_token_present,
        "stop_tokens_suspicious": result.stop_tokens_suspicious,
        "retrieval_injection_risk": result.retrieval_injection_risk,
        "chat_template_confidence": result.chat_template_confidence,
        "risk_flags": result.risk_flags,
        "chat_format": chat_format,
        "merged_stop_count": merged_stop_count,
        "reconstruction_note": reconstruction_note,
        "model_reasoning_profile_detected": result.model_reasoning_profile_detected,
        "execution_mode": result.execution_mode,
    }
    if parity is not None:
        payload["parity_score"] = parity.score
        payload["parity_differences"] = parity.differences
        payload["parity_likely_root_cause"] = parity.likely_root_cause
    logger.info(json.dumps(payload, ensure_ascii=False))


def validation_result_to_dict(result: PromptValidationResult) -> dict[str, Any]:
    return asdict(result)
