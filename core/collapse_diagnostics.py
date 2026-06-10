"""
Per-turn collapse diagnostics for model degradation onset detection.

Computes prompt/output metrics, rewrite/degeneration signals, hallucination and
format-drift scores, and emits ``collapse_risk`` (LOW|MEDIUM|HIGH) for traces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from core.history_degeneration import score_history_degeneration
from core.output_validation import validate_output
from core.prompt_contract import PromptContract
from core.response_quality import evaluate_response_quality

CollapseRisk = Literal["LOW", "MEDIUM", "HIGH"]

_DUMMY_CONTRACT = PromptContract(
    mode="messages",
    chat_format="chatml",
    prompt=None,
    messages=[{"role": "user", "content": "hi"}],
    stop=[],
    template_source="fallback",
    confidence="medium",
)

_ORPHAN_WEB_CITATION = re.compile(r"\[W\]", re.I)
_BARE_CITATION_ONLY = re.compile(r"^\s*\[\d+\]\s*$")
_HARMONY_TOKEN = re.compile(r"<\|(?:channel|message|final|start)\|>", re.I)
_BROKEN_LIST_TAIL = re.compile(r"\n\d+\.\s*(?:\*\*[^*\n]*)?$")
_META_APOLOGY = re.compile(
    r"(?i)\b(?:sorry|just kidding|i mean|my apologies|never mind|scratch that)\b"
)

_FORMAT_DRIFT_WEIGHTS: dict[str, float] = {
    "template_leakage": 0.55,
    "harmony_no_final_answer": 0.50,
    "meta_preamble": 0.45,
    "role_confusion": 0.50,
    "truncated_output": 0.35,
    "degeneration": 0.40,
    "harmony_token_leak": 0.55,
    "broken_markdown_list": 0.40,
}

_HALLUCINATION_WEIGHTS: dict[str, float] = {
    "low_relevance": 0.35,
    "retrieval_ignored": 0.25,
    "incomplete_answer": 0.30,
    "coherence_issue": 0.35,
    "overconfident_claim_signal": 0.30,
    "low_utility": 0.20,
    "constraint_missed_brief": 0.15,
    "constraint_missed_list": 0.15,
    "orphan_web_citation": 0.45,
    "bare_citation_token": 0.50,
    "meta_apology": 0.25,
    "denies_established_fact": 0.55,
    "prior_turn_suppressed_confabulation": 0.40,
}

_UNESCO_QUERY = re.compile(r"\b(unesco|world heritage)\b", re.I)
_DENIES_PRESENCE = re.compile(
    r"(?i)(?:"
    r"\bthere are no\b.{0,80}\b(?:unesco|world heritage|heritage sites?)\b"
    r"|"
    r"\bno unesco\b"
    r"|"
    r"\bnone (?:are )?located within\b"
    r"|"
    r"\bnot located within\b.{0,60}\b(?:city|capital)\b"
    r")"
)
_LISTING_QUERY = re.compile(
    r"(?i)\b(what|which|list|name|major|main)\b.{0,60}\b("
    r"attractions|sites|heritage|landmarks|monuments"
    r")\b"
)

_PROMPT_STRESS_CHARS = 12_000
_PROMPT_WARN_CHARS = 8_000


@dataclass(frozen=True)
class CollapseTurnDiagnostics:
    turn_index: int
    prompt_length: int
    output_length: int
    rewrite_confidence: float
    degeneration_score: float
    hallucination_score: float
    format_drift_score: float
    hallucination_flags: tuple[str, ...]
    format_drift_flags: tuple[str, ...]
    collapse_score: float
    collapse_risk: CollapseRisk
    prior_turn_suppressed: bool = False

    def trace_fields(self) -> dict[str, object]:
        fields = {
            "collapse_turn_index": self.turn_index,
            "collapse_prompt_length": self.prompt_length,
            "collapse_output_length": self.output_length,
            "collapse_rewrite_confidence": round(self.rewrite_confidence, 3),
            "collapse_degeneration_score": round(self.degeneration_score, 3),
            "collapse_hallucination_score": round(self.hallucination_score, 3),
            "collapse_format_drift_score": round(self.format_drift_score, 3),
            "collapse_hallucination_flags": list(self.hallucination_flags),
            "collapse_format_drift_flags": list(self.format_drift_flags),
            "collapse_score": round(self.collapse_score, 3),
            "collapse_risk": self.collapse_risk,
        }
        if self.prior_turn_suppressed:
            fields["prior_turn_suppressed"] = True
        return fields

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        turn_index: int = 0,
    ) -> CollapseTurnDiagnostics:
        meta = metadata or {}
        risk_raw = str(meta.get("collapse_risk") or "LOW").upper()
        risk: CollapseRisk = (
            risk_raw if risk_raw in ("LOW", "MEDIUM", "HIGH") else "LOW"
        )
        return cls(
            turn_index=int(meta.get("collapse_turn_index", turn_index)),
            prompt_length=int(meta.get("collapse_prompt_length", 0)),
            output_length=int(meta.get("collapse_output_length", 0)),
            rewrite_confidence=float(meta.get("collapse_rewrite_confidence", 0.0)),
            degeneration_score=float(
                meta.get(
                    "collapse_degeneration_score",
                    meta.get("history_degeneration_score", 0.0),
                )
            ),
            hallucination_score=float(meta.get("collapse_hallucination_score", 0.0)),
            format_drift_score=float(meta.get("collapse_format_drift_score", 0.0)),
            hallucination_flags=tuple(meta.get("collapse_hallucination_flags") or ()),
            format_drift_flags=tuple(meta.get("collapse_format_drift_flags") or ()),
            collapse_score=float(meta.get("collapse_score", 0.0)),
            collapse_risk=risk,
            prior_turn_suppressed=bool(meta.get("prior_turn_suppressed")),
        )


def score_format_drift(output: str) -> tuple[float, tuple[str, ...]]:
    text = (output or "").strip()
    if not text:
        return 0.0, ()

    flags: list[str] = []
    validation = validate_output(text, _DUMMY_CONTRACT)
    for issue in validation.issues:
        if issue not in flags:
            flags.append(issue)

    if _HARMONY_TOKEN.search(text) and "harmony_token_leak" not in flags:
        flags.append("harmony_token_leak")

    if _BROKEN_LIST_TAIL.search(text) and "broken_markdown_list" not in flags:
        flags.append("broken_markdown_list")

    score = min(
        1.0,
        sum(_FORMAT_DRIFT_WEIGHTS.get(flag, 0.25) for flag in flags),
    )
    return round(score, 4), tuple(flags)


def score_hallucination_indicators(
    *,
    user_query: str,
    output: str,
    prior_turn_suppressed: bool = False,
    active_referent: str = "",
) -> tuple[float, tuple[str, ...]]:
    text = (output or "").strip()
    if not text:
        return 0.0, ()

    flags: list[str] = []
    quality = evaluate_response_quality(user_query, text)
    for issue in quality.issues:
        if issue not in flags:
            flags.append(issue)

    if _ORPHAN_WEB_CITATION.search(text) and "orphan_web_citation" not in flags:
        flags.append("orphan_web_citation")

    if _BARE_CITATION_ONLY.match(text) and "bare_citation_token" not in flags:
        flags.append("bare_citation_token")

    if _META_APOLOGY.search(text) and "meta_apology" not in flags:
        flags.append("meta_apology")

    query = (user_query or "").strip()
    if _UNESCO_QUERY.search(query) and _DENIES_PRESENCE.search(text):
        flags.append("denies_established_fact")

    if (
        _LISTING_QUERY.search(query)
        and _DENIES_PRESENCE.search(text)
        and "denies_established_fact" not in flags
    ):
        flags.append("denies_established_fact")

    if prior_turn_suppressed and len(text) >= 40:
        flags.append("prior_turn_suppressed_confabulation")

    _ = active_referent  # reserved for future referent-aware checks

    score = min(
        1.0,
        sum(_HALLUCINATION_WEIGHTS.get(flag, 0.20) for flag in flags),
    )
    return round(score, 4), tuple(flags)


def _prompt_stress(prompt_length: int) -> float:
    if prompt_length >= _PROMPT_STRESS_CHARS:
        return 0.35
    if prompt_length >= _PROMPT_WARN_CHARS:
        return 0.18
    if prompt_length >= 6_000:
        return 0.08
    return 0.0


def _rewrite_penalty(rewrite_confidence: float) -> float:
    if rewrite_confidence <= 0.0:
        return 0.0
    return max(0.0, min(0.35, 0.75 - rewrite_confidence))


def _resolve_collapse_risk(
    *,
    collapse_score: float,
    degeneration_score: float,
    hallucination_score: float,
    format_drift_score: float,
    hallucination_flags: tuple[str, ...] = (),
    prior_turn_suppressed: bool = False,
) -> CollapseRisk:
    if (
        degeneration_score >= 0.50
        or collapse_score >= 0.65
        or (hallucination_score >= 0.55 and format_drift_score >= 0.35)
        or (
            prior_turn_suppressed
            and "denies_established_fact" in hallucination_flags
        )
    ):
        return "HIGH"
    if (
        collapse_score >= 0.38
        or degeneration_score >= 0.30
        or hallucination_score >= 0.40
        or format_drift_score >= 0.45
        or (
            prior_turn_suppressed
            and (
                "prior_turn_suppressed_confabulation" in hallucination_flags
                or hallucination_score >= 0.35
            )
        )
    ):
        return "MEDIUM"
    return "LOW"


def compute_collapse_diagnostics(
    *,
    prompt: str,
    output: str,
    user_query: str = "",
    rewrite_confidence: float = 0.0,
    degeneration_score: float | None = None,
    degeneration_flags: tuple[str, ...] = (),
    turn_index: int = 0,
    prior_turn_suppressed: bool = False,
    active_referent: str = "",
) -> CollapseTurnDiagnostics:
    prompt_text = prompt or ""
    output_text = output or ""
    prompt_length = len(prompt_text)
    output_length = len(output_text)

    if degeneration_score is None:
        degeneration = score_history_degeneration(output_text)
        degeneration_score = degeneration.score
        degeneration_flags = degeneration.flags

    format_drift_score, format_flags = score_format_drift(output_text)
    hallucination_score, hallucination_flags = score_hallucination_indicators(
        user_query=user_query or "",
        output=output_text,
        prior_turn_suppressed=prior_turn_suppressed,
        active_referent=active_referent,
    )

    collapse_score = min(
        1.0,
        round(
            0.30 * float(degeneration_score)
            + 0.25 * format_drift_score
            + 0.25 * hallucination_score
            + 0.10 * _rewrite_penalty(rewrite_confidence)
            + 0.10 * _prompt_stress(prompt_length),
            4,
        ),
    )
    collapse_risk = _resolve_collapse_risk(
        collapse_score=collapse_score,
        degeneration_score=float(degeneration_score),
        hallucination_score=hallucination_score,
        format_drift_score=format_drift_score,
        hallucination_flags=hallucination_flags,
        prior_turn_suppressed=prior_turn_suppressed,
    )

    _ = degeneration_flags  # retained for future weighting; flags live in history metadata

    return CollapseTurnDiagnostics(
        turn_index=turn_index,
        prompt_length=prompt_length,
        output_length=output_length,
        rewrite_confidence=float(rewrite_confidence),
        degeneration_score=float(degeneration_score),
        hallucination_score=hallucination_score,
        format_drift_score=format_drift_score,
        hallucination_flags=hallucination_flags,
        format_drift_flags=format_flags,
        collapse_score=collapse_score,
        collapse_risk=collapse_risk,
        prior_turn_suppressed=prior_turn_suppressed,
    )


def diagnostics_from_trace_metadata(
    *,
    metadata: dict[str, Any] | None,
    prompt: str,
    output: str,
    user_query: str = "",
    turn_index: int = 0,
    prior_turn_suppressed: bool = False,
    active_referent: str = "",
) -> CollapseTurnDiagnostics:
    """Reuse stored collapse fields when present; otherwise compute post-hoc."""
    meta = metadata or {}
    if meta.get("collapse_risk") and not prior_turn_suppressed:
        return CollapseTurnDiagnostics.from_metadata(meta, turn_index=turn_index)

    rewrite_confidence = float(meta.get("rewrite_confidence", 0.0) or 0.0)
    degeneration_score = meta.get("history_degeneration_score")
    degeneration_flags: tuple[str, ...] = tuple(
        meta.get("history_degeneration_flags") or ()
    )
    prior = prior_turn_suppressed or bool(meta.get("prior_turn_suppressed"))

    return compute_collapse_diagnostics(
        prompt=prompt,
        output=output,
        user_query=user_query,
        rewrite_confidence=rewrite_confidence,
        degeneration_score=(
            float(degeneration_score) if degeneration_score is not None else None
        ),
        degeneration_flags=degeneration_flags,
        turn_index=turn_index,
        prior_turn_suppressed=prior,
        active_referent=active_referent,
    )


def build_collapse_timeline(
    turns: list[Any],
    *,
    backend_label: str = "",
) -> list[dict[str, object]]:
    """
    Build a serial timeline from ``TurnTrace`` objects or dicts with trace metadata.

    Each entry is JSON-safe for UI rendering and scenario diff artifacts.
    """
    timeline: list[dict[str, object]] = []
    prior_suppressed = False
    for turn in turns or []:
        turn_index = int(getattr(turn, "turn_index", turn.get("turn_index", 0)))
        user_message = str(
            getattr(turn, "user_message", turn.get("user_message", "")) or ""
        )
        trace_obj = getattr(turn, "trace", turn.get("trace"))
        if trace_obj is None:
            continue
        prompt = str(getattr(trace_obj, "prompt", trace_obj.get("prompt", "")) or "")
        output = str(getattr(trace_obj, "output", trace_obj.get("output", "")) or "")
        metadata = getattr(trace_obj, "metadata", trace_obj.get("metadata", {})) or {}

        diag = diagnostics_from_trace_metadata(
            metadata=metadata if isinstance(metadata, dict) else {},
            prompt=prompt,
            output=output,
            user_query=user_message,
            turn_index=turn_index,
            prior_turn_suppressed=prior_suppressed,
        )
        entry = {
            "backend": backend_label,
            **diag.trace_fields(),
            "user_message_preview": user_message[:80],
        }
        timeline.append(entry)
        prior_suppressed = bool(
            (metadata if isinstance(metadata, dict) else {}).get(
                "history_degeneration_suppressed"
            )
        )
    return timeline
