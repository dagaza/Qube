"""
Turn-level generation debug capture for collapse investigations.

Enable with ``QUBE_GENERATION_DEBUG=1``. Artifacts land under ``debug_generation/``
(override with ``QUBE_GENERATION_DEBUG_DIR``).

Optional env:
  ``QUBE_GENERATION_DEBUG_RUN`` — subfolder suffix (e.g. ``temp0.3``, ``stops_minimal``)
  ``QUBE_GENERATION_DEBUG_TURNS`` — comma list of 1-based turn ids (default: all turns)
  ``QUBE_GENERATION_DEBUG_TEMPERATURE`` — override sampling temperature for the run
  ``QUBE_GENERATION_DEBUG_STOP_MODE`` — ``full`` (default) or ``minimal``
  ``QUBE_LLM_TOKEN_TRACE=1`` — include sampler ground-truth token ids in trace JSON

Observer only — does not alter prompts or model output.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.collapse_diagnostics import compute_collapse_diagnostics, score_format_drift
from core.completion_output_trace import CompletionOutputSnapshot
from core.harmony_protocol import HARMONY_PRIMARY_STOPS
from core.history_degeneration import score_history_degeneration

logger = logging.getLogger("Qube.GenerationDebug")

_MALFORMED_LIST = re.compile(
    r"(?:\n\d+\.\s*\n|\*\*\s*\n|\[\d+\]\s*$|<\|(?:channel|message|start)\|>)"
)
_TOKEN_SOUP = re.compile(r"(?:<\|[^>]+\|>|\[\d+\]\s*\[\d+\])")


def generation_debug_enabled() -> bool:
    return os.environ.get("QUBE_GENERATION_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def generation_debug_stop_mode() -> str:
    return (os.environ.get("QUBE_GENERATION_DEBUG_STOP_MODE") or "full").strip().lower()


def generation_debug_temperature_override() -> Optional[float]:
    raw = (os.environ.get("QUBE_GENERATION_DEBUG_TEMPERATURE") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def apply_debug_sampling_overrides(gen_params: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with debug temperature override applied."""
    out = dict(gen_params)
    temp = generation_debug_temperature_override()
    if temp is not None:
        out["temperature"] = temp
    return out


def apply_debug_stop_mode(merged_stops: list[str], eos_token: str = "") -> list[str]:
    """When ``QUBE_GENERATION_DEBUG_STOP_MODE=minimal``, keep primary Harmony/EOS stops only."""
    if generation_debug_stop_mode() != "minimal":
        return list(merged_stops or [])
    out: list[str] = []
    for stop in HARMONY_PRIMARY_STOPS:
        if stop and stop not in out:
            out.append(stop)
    if eos_token and eos_token not in out:
        out.append(eos_token)
    return out or list(merged_stops or [])


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generation_debug_output_dir() -> str:
    base = (os.environ.get("QUBE_GENERATION_DEBUG_DIR") or "").strip()
    if not base:
        base = os.path.join(_repo_root(), "debug_generation")
    run_suffix = (os.environ.get("QUBE_GENERATION_DEBUG_RUN") or "").strip()
    if run_suffix:
        base = os.path.join(base, run_suffix)
    os.makedirs(base, exist_ok=True)
    return base


def _parse_turn_filter() -> Optional[set[int]]:
    raw = (os.environ.get("QUBE_GENERATION_DEBUG_TURNS") or "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out or None


def should_capture_turn(turn_id: int) -> bool:
    if not generation_debug_enabled():
        return False
    filt = _parse_turn_filter()
    if filt is None:
        return True
    return int(turn_id) in filt


@dataclass
class ChunkRecord:
    chunk_index: int
    delta: str
    cumulative_raw: str
    cumulative_filtered: str
    token_id: Optional[int] = None
    logit_probability: Optional[float] = None
    events: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationDebugRecorder:
    turn_id: int
    session_id: str = ""
    user_query: str = ""
    chunks: list[ChunkRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    _gt_token_ids: list[int] = field(default_factory=list)
    _gt_token_texts: list[str] = field(default_factory=list)
    _finish_reason: str = ""
    _stream_cancel_reason: str = ""
    _final_stored: str = ""
    _snapshot: Optional[CompletionOutputSnapshot] = None

    @classmethod
    def maybe_start(
        cls,
        *,
        turn_id: int,
        session_id: str = "",
        user_query: str = "",
        gen_params: dict[str, Any] | None = None,
        native_preflight: dict[str, Any] | None = None,
        merged_stops: list[str] | None = None,
    ) -> Optional["GenerationDebugRecorder"]:
        if not should_capture_turn(turn_id):
            return None
        pre = native_preflight or {}
        sampling = dict(pre.get("sampling_snapshot") or {})
        if gen_params:
            sampling.setdefault("temperature", gen_params.get("temperature"))
            sampling.setdefault("max_tokens", gen_params.get("max_tokens"))
            overrides = gen_params.get("sampling_overrides") or {}
            if isinstance(overrides, dict):
                for key in ("top_k", "top_p", "repeat_penalty", "min_p", "presence_penalty"):
                    if key in overrides:
                        sampling[key] = overrides[key]
        rec = cls(
            turn_id=int(turn_id),
            session_id=str(session_id or ""),
            user_query=str(user_query or ""),
        )
        rec.meta = {
            "turn_id": int(turn_id),
            "session_id": rec.session_id,
            "user_query_preview": rec.user_query[:120],
            "temperature": sampling.get("temperature"),
            "top_k": sampling.get("top_k"),
            "top_p": sampling.get("top_p"),
            "repeat_penalty": sampling.get("repeat_penalty"),
            "min_p": sampling.get("min_p"),
            "presence_penalty": sampling.get("presence_penalty"),
            "max_tokens": sampling.get("max_tokens"),
            "stop_tokens": list(merged_stops or pre.get("merged_stops") or []),
            "stop_token_count": len(merged_stops or pre.get("merged_stops") or []),
            "eos_token_str": pre.get("eos_token_str", ""),
            "debug_stop_mode": generation_debug_stop_mode(),
            "debug_temperature_override": generation_debug_temperature_override(),
            "debug_run": os.environ.get("QUBE_GENERATION_DEBUG_RUN", ""),
            "finish_reason": "",
            "stream_cancel_reason": "",
            "logits_available": False,
            "logits_note": (
                "Per-token logit probabilities are not captured by default. "
                "Enable llama.cpp logprobs in native engine if needed."
            ),
        }
        return rec

    def record_delta(
        self,
        *,
        delta: str,
        cumulative_raw: str,
        cumulative_filtered: str,
        events: dict[str, Any] | None = None,
    ) -> None:
        idx = len(self.chunks)
        token_id: Optional[int] = None
        if self._gt_token_ids and idx < len(self._gt_token_ids):
            token_id = self._gt_token_ids[idx]
        self.chunks.append(
            ChunkRecord(
                chunk_index=idx,
                delta=delta or "",
                cumulative_raw=cumulative_raw or "",
                cumulative_filtered=cumulative_filtered or "",
                token_id=token_id,
                logit_probability=None,
                events=dict(events or {}),
            )
        )

    def note_stream_cancel(self, reason: str) -> None:
        self._stream_cancel_reason = str(reason or "")

    def finalize_stream(
        self,
        *,
        snapshot: CompletionOutputSnapshot,
        gt_token_ids: list[int] | None = None,
        gt_token_texts: list[str] | None = None,
        finish_reason: str = "",
        native_preflight: dict[str, Any] | None = None,
        merged_stops: list[str] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._gt_token_ids = list(gt_token_ids or [])
        self._gt_token_texts = list(gt_token_texts or [])
        self._finish_reason = str(finish_reason or self._stream_cancel_reason or "complete")
        if native_preflight:
            sampling = dict(native_preflight.get("sampling_snapshot") or {})
            for key in ("temperature", "max_tokens", "top_p", "repeat_penalty"):
                if key in sampling:
                    self.meta[key] = sampling[key]
            self.meta["eos_token_str"] = native_preflight.get("eos_token_str", "")
        if merged_stops is not None:
            self.meta["stop_tokens"] = list(merged_stops)
            self.meta["stop_token_count"] = len(merged_stops)
        self.meta["finish_reason"] = self._finish_reason
        self.meta["stream_cancel_reason"] = self._stream_cancel_reason
        self.meta["ground_truth_token_count"] = len(self._gt_token_ids)
        if self._gt_token_ids:
            self.meta["ground_truth_token_ids_sample"] = self._gt_token_ids[:64]

    def record_final_stored(self, stored_content: str, *, ui_final: str = "") -> None:
        self._final_stored = stored_content or ""
        self.meta["history_suppressed"] = stored_content != (ui_final or stored_content)
        self.write_artifacts(ui_final=ui_final or stored_content)

    def write_artifacts(self, *, ui_final: str = "") -> None:
        out_dir = generation_debug_output_dir()
        prefix = f"turn{self.turn_id}"
        raw_path = os.path.join(out_dir, f"{prefix}_raw_stream.txt")
        post_path = os.path.join(out_dir, f"{prefix}_postprocess.txt")
        final_path = os.path.join(out_dir, f"{prefix}_final.txt")
        meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
        trace_path = os.path.join(out_dir, f"{prefix}_trace_analysis.json")

        with open(raw_path, "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(f"=== chunk {chunk.chunk_index} ===\n")
                fh.write(f"delta: {chunk.delta!r}\n")
                fh.write(f"cumulative_raw_len: {len(chunk.cumulative_raw)}\n")
                if chunk.token_id is not None:
                    fh.write(f"token_id: {chunk.token_id}\n")
                if chunk.events:
                    fh.write(f"events: {json.dumps(chunk.events, ensure_ascii=False)}\n")
                fh.write(f"cumulative_raw:\n{chunk.cumulative_raw}\n\n")
            if self._gt_token_texts:
                fh.write("\n=== sampler_ground_truth (first tokens) ===\n")
                fh.write("".join(self._gt_token_texts[:256]))
                fh.write("\n")

        snap = self._snapshot
        post_stages: dict[str, str] = {}
        if snap is not None:
            post_stages = {
                "raw_text": snap.raw_text or "",
                "after_harmony_parser": snap.after_harmony_parser or "",
                "after_worker_filters": snap.after_worker_filters or "",
                "streamed_incremental": snap.streamed_incremental or "",
                "worker_return_text": snap.worker_return_text or "",
                "engine_end_text": snap.engine_end_text or "",
            }
        with open(post_path, "w", encoding="utf-8") as fh:
            for name, text in post_stages.items():
                fh.write(f"--- {name} (len={len(text)}) ---\n")
                fh.write(text)
                fh.write("\n\n")

        final_text = self._final_stored or ui_final or ""
        with open(final_path, "w", encoding="utf-8") as fh:
            fh.write(final_text)

        origin = analyze_corruption_origin(
            raw=post_stages.get("raw_text", ""),
            after_harmony=post_stages.get("after_harmony_parser", ""),
            after_filters=post_stages.get("after_worker_filters", ""),
            worker_return=post_stages.get("worker_return_text", ""),
            stored=final_text,
        )
        self.meta["corruption_origin"] = origin
        collapse = compute_collapse_diagnostics(
            prompt="",
            output=final_text,
            user_query=self.user_query,
            turn_index=self.turn_id,
        )
        self.meta["collapse_risk"] = collapse.collapse_risk
        self.meta["collapse_score"] = collapse.collapse_score
        self.meta["format_drift_flags"] = list(collapse.format_drift_flags)

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, ensure_ascii=False)

        trace_rows = []
        for chunk in self.chunks:
            row: dict[str, Any] = {
                "chunk_index": chunk.chunk_index,
                "token": chunk.delta,
                "cumulative_text": chunk.cumulative_raw,
                "cumulative_filtered": chunk.cumulative_filtered,
                "token_id": chunk.token_id,
                "logit_probability": chunk.logit_probability,
                "early_stop_triggers": chunk.events.get("early_stop_triggers", []),
                "repair_triggers": chunk.events.get("repair_triggers", []),
                "guard_events": chunk.events.get("guard_events", []),
            }
            trace_rows.append(row)

        trace_payload = {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "chunk_count": len(trace_rows),
            "corruption_origin": origin,
            "chunks": trace_rows,
        }
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(trace_payload, fh, indent=2, ensure_ascii=False)

        logger.info(
            "[GenerationDebug] wrote turn %s artifacts to %s (collapse_risk=%s origin=%s)",
            self.turn_id,
            out_dir,
            collapse.collapse_risk,
            origin.get("likely_stage"),
        )


def analyze_corruption_origin(
    *,
    raw: str,
    after_harmony: str,
    after_filters: str,
    worker_return: str,
    stored: str,
) -> dict[str, Any]:
    """Compare pipeline stages to locate where malformed output first appears."""

    stages = [
        ("raw_generation", raw),
        ("after_harmony_parser", after_harmony),
        ("after_worker_filters", after_filters),
        ("worker_return", worker_return),
        ("history_stored", stored),
    ]

    def _signals(text: str) -> dict[str, Any]:
        t = text or ""
        fmt_score, fmt_flags = score_format_drift(t)
        hist = score_history_degeneration(t)
        return {
            "len": len(t),
            "malformed_list": bool(_MALFORMED_LIST.search(t)),
            "token_soup": bool(_TOKEN_SOUP.search(t)),
            "format_drift_score": fmt_score,
            "format_drift_flags": list(fmt_flags),
            "degeneration_score": hist.score,
            "degeneration_flags": list(hist.flags),
        }

    stage_signals = {name: _signals(text) for name, text in stages}

    def _bad(sig: dict[str, Any]) -> bool:
        return (
            sig["malformed_list"]
            or sig["token_soup"]
            or sig["format_drift_score"] >= 0.35
            or sig["degeneration_score"] >= 0.35
        )

    first_bad: Optional[str] = None
    for name, _ in stages:
        if _bad(stage_signals[name]):
            first_bad = name
            break

    likely_stage = first_bad or "none_detected"
    likely_cause = "unknown"
    if first_bad == "raw_generation":
        likely_cause = "sampling_or_model_generation"
    elif first_bad in ("after_harmony_parser", "after_worker_filters"):
        likely_cause = "post_processing_or_harmony_parser"
    elif first_bad == "worker_return":
        likely_cause = "worker_stream_assembly"
    elif first_bad == "history_stored":
        likely_cause = "history_degeneration_suppression"

    raw_sig = stage_signals["raw_generation"]
    stored_sig = stage_signals["history_stored"]
    postprocess_introduced = (
        not _bad(raw_sig)
        and _bad(stage_signals["after_harmony_parser"])
    ) or (
        stage_signals["raw_generation"]["len"] == stage_signals["after_worker_filters"]["len"]
        and _bad(stored_sig)
        and not _bad(stage_signals["after_worker_filters"])
    )

    return {
        "likely_stage": likely_stage,
        "likely_cause": likely_cause,
        "postprocess_introduced_corruption": postprocess_introduced,
        "stage_signals": stage_signals,
        "raw_equals_stored": (raw or "") == (stored or ""),
        "raw_equals_worker_return": (raw or "") == (worker_return or ""),
    }


def build_diagnostic_summary(output_dir: str | None = None) -> dict[str, Any]:
    """Scan ``debug_generation/`` turn artifacts and emit a structured summary."""
    base = output_dir or generation_debug_output_dir()
    if not os.path.isdir(base):
        return {"error": f"directory not found: {base}"}

    turns: list[dict[str, Any]] = []
    turn_ids: list[int] = []
    for name in os.listdir(base):
        if not name.endswith("_meta.json") or not name.startswith("turn"):
            continue
        try:
            tid = int(name.replace("turn", "").replace("_meta.json", ""))
        except ValueError:
            continue
        turn_ids.append(tid)
    turn_ids.sort()

    first_collapse_turn: Optional[int] = None
    for tid in turn_ids:
        meta_path = os.path.join(base, f"turn{tid}_meta.json")
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        risk = str(meta.get("collapse_risk") or "LOW").upper()
        origin = meta.get("corruption_origin") or {}
        entry = {
            "turn_id": tid,
            "collapse_risk": risk,
            "collapse_score": meta.get("collapse_score"),
            "likely_cause": origin.get("likely_cause"),
            "likely_stage": origin.get("likely_stage"),
            "finish_reason": meta.get("finish_reason"),
            "stream_cancel_reason": meta.get("stream_cancel_reason"),
            "temperature": meta.get("temperature"),
            "stop_token_count": meta.get("stop_token_count"),
        }
        turns.append(entry)
        if first_collapse_turn is None and risk in ("MEDIUM", "HIGH"):
            first_collapse_turn = tid

    # Recommend next step from aggregate pattern
    causes = [t.get("likely_cause") for t in turns if t.get("likely_cause")]
    dominant_cause = max(set(causes), key=causes.count) if causes else "unknown"
    recommendations: list[str] = []
    if dominant_cause == "sampling_or_model_generation":
        recommendations.append(
            "Replay at temperature 0.0–0.3 and compare raw_stream; if collapse persists, "
            "inspect prompt length and context window pressure."
        )
        recommendations.append(
            "Run stop-mode A/B (full vs minimal) to rule out premature stop-token hits."
        )
    elif dominant_cause == "post_processing_or_harmony_parser":
        recommendations.append(
            "Compare raw_stream vs postprocess stages; disable harmony parser incrementally in a test branch."
        )
    elif dominant_cause == "history_degeneration_suppression":
        recommendations.append(
            "Inspect prior-turn history placeholders and degeneration scores feeding turn context."
        )
    else:
        recommendations.append(
            "Collect turns 4–7 with QUBE_LLM_TOKEN_TRACE=1 and compare sampler ground truth to raw_stream."
        )

    summary = {
        "output_dir": base,
        "turn_count": len(turns),
        "first_collapse_turn": first_collapse_turn,
        "dominant_likely_cause": dominant_cause,
        "recommended_next_steps": recommendations,
        "turns": turns,
    }
    summary_path = os.path.join(base, "diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary
