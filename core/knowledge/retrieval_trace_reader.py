"""Read retrieval traces from the web search audit JSONL log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.knowledge.observability import CITATION_GROUNDING_EVENT, RETRIEVAL_TRACE_EVENT
from core.knowledge.search_outcome import (
    format_search_outcome_summary_line,
    search_outcome_from_relevance_diag,
)
from core.web_search_audit_sink import default_web_search_audit_log_path


def _parse_event(line: str, event: str) -> dict[str, Any] | None:
    line = (line or "").strip()
    if not line:
        return None
    # The file sink writes "[time] [LEVEL] [logger] {json}"; tests and older
    # tools write bare JSON. Parse from the first brace either way (S11 fix:
    # before it, no line written by the real sink could be read).
    brace = line.find("{")
    if brace > 0:
        line = line[brace:]
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("event") != event:
        return None
    return payload


def _parse_line(line: str) -> dict[str, Any] | None:
    return _parse_event(line, RETRIEVAL_TRACE_EVENT)


def _turn_key(payload: dict[str, Any]) -> tuple[str, str] | None:
    sid = str(payload.get("session_id") or "")
    tid = payload.get("turn_id")
    if not sid or tid is None:
        return None
    return sid, str(tid)


def read_retrieval_traces(
    *,
    log_path: Path | None = None,
    limit: int = 50,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return retrieval_trace events newest-first."""
    path = log_path or default_web_search_audit_log_path()
    if not path.is_file():
        return []

    traces: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    # Plan S11: post-stream grounding events, newest per (session, turn).
    grounding: dict[tuple[str, str], dict[str, Any]] = {}
    for line in reversed(lines):
        if CITATION_GROUNDING_EVENT not in line:
            continue
        event = _parse_event(line, CITATION_GROUNDING_EVENT)
        key = _turn_key(event) if event is not None else None
        if key is not None and key not in grounding:
            grounding[key] = event

    for line in reversed(lines):
        payload = _parse_line(line)
        if payload is None:
            continue
        key = _turn_key(payload)
        if key is not None and key in grounding:
            payload["citation_grounding"] = grounding[key]
        if session_id and str(payload.get("session_id") or "") != session_id:
            continue
        if turn_id is not None:
            try:
                if int(payload.get("turn_id")) != turn_id:
                    continue
            except (TypeError, ValueError):
                continue
        traces.append(payload)
        if len(traces) >= max(1, limit):
            break
    return traces


def read_last_retrieval_trace(
    *,
    log_path: Path | None = None,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> dict[str, Any] | None:
    traces = read_retrieval_traces(
        log_path=log_path,
        limit=1,
        session_id=session_id,
        turn_id=turn_id,
    )
    return traces[0] if traces else None


_CONSIDERED_NOT_USED_LIMIT = 15


def format_relevance_gate_lines(trace: dict[str, Any]) -> list[str]:
    """Relevance gate summary plus "Considered, not used" (plan S4, decision D2).

    Rows come from ``relevance_diag["rerank_dropped"]``: sources the gate scored
    below its threshold. They are shown here only — never in the sources
    panel, the bundle, the prompt or citation numbering.
    """
    diag = trace.get("relevance_diag") or {}
    rerank = diag.get("rerank") if isinstance(diag, dict) else None
    sufficiency = str(trace.get("sufficiency") or diag.get("sufficiency") or "")
    lines: list[str] = []
    if isinstance(rerank, dict) and rerank:
        skipped = rerank.get("skipped_reason")
        if skipped:
            lines.append(f"Relevance gate: not applied ({skipped}); legacy ranking used")
        else:
            lines.append(
                f"Relevance gate: {sufficiency or rerank.get('state') or '—'} — "
                f"{rerank.get('scored', '—')} scored, {rerank.get('survivors', '—')} passed "
                f"(threshold {rerank.get('threshold_keep', '—')}), "
                f"{rerank.get('latency_ms', '—')} ms"
            )
    dropped = diag.get("rerank_dropped") if isinstance(diag, dict) else None
    if isinstance(dropped, list) and dropped:
        lines.append(f"Considered, not used ({len(dropped)}):")
        for row in dropped[:_CONSIDERED_NOT_USED_LIMIT]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "(untitled)")[:100]
            adapter = row.get("adapter")
            score = row.get("score")
            score_txt = f"{float(score):.2f}" if isinstance(score, (int, float)) else "—"
            suffix = f" [{adapter}]" if adapter else ""
            lines.append(f"  - {title}{suffix} — relevance {score_txt}")
        if len(dropped) > _CONSIDERED_NOT_USED_LIMIT:
            lines.append(f"  … {len(dropped) - _CONSIDERED_NOT_USED_LIMIT} more")
    return lines


def format_citation_grounding_line(event: dict[str, Any]) -> str:
    """One line on post-stream citation grounding (plan S11; telemetry only)."""
    if not event:
        return ""
    if event.get("status") != "scored":
        reason = event.get("skip_reason") or "not scored"
        return f"Citation grounding: not scored ({reason})"
    return (
        "Citation grounding (relevance proxy, not a verdict): "
        f"{event.get('claims', 0)} cited sentence(s), {event.get('pairs_scored', 0)} scored; "
        f"median {event.get('score_median')}, min {event.get('score_min')}; "
        f"{event.get('claims_best_below_0', 0)} with best source below 0"
    )


def format_retrieval_trace_summary(trace: dict[str, Any]) -> str:
    """Human-readable summary for UI display."""
    if not trace:
        return "No retrieval trace available."

    lines = [
        f"Service: {trace.get('knowledge_service', '—')}",
        f"Strategy: {trace.get('retrieval_strategy', '—')}",
        f"Adapters: {', '.join(trace.get('adapter_calls') or []) or '—'}",
        f"Sources kept: {len(trace.get('evidence_ids_kept') or [])}",
        f"Coverage: {trace.get('coverage', '—')} (confidence {trace.get('confidence', '—')})",
        f"Latency: {trace.get('latency_ms', '—')} ms",
        f"Stop reason: {trace.get('stop_reason', '—')}",
    ]
    if trace.get("request_id"):
        lines.append(f"Request id: {trace.get('request_id')}")
    if trace.get("bundle_id"):
        lines.append(f"Bundle id: {trace.get('bundle_id')}")
    if trace.get("preset_id"):
        lines.append(f"Preset id: {trace.get('preset_id')}")
    if trace.get("retrieval_profile"):
        lines.append(f"Retrieval profile: {trace.get('retrieval_profile')}")
    cap_steps = trace.get("capability_steps") or []
    if cap_steps:
        from core.integrations.capability_inspect import format_capability_steps_summary_line

        summary = format_capability_steps_summary_line(cap_steps)
        if summary:
            lines.append(summary)
    search_line = format_search_outcome_summary_line(
        search_outcome_from_relevance_diag(trace.get("relevance_diag") or {})
    )
    if search_line:
        lines.append(search_line)
    lines.extend(format_relevance_gate_lines(trace))
    grounding_line = format_citation_grounding_line(trace.get("citation_grounding") or {})
    if grounding_line:
        lines.append(grounding_line)
    warnings = trace.get("warnings") or []
    if warnings:
        lines.append(f"Warnings: {'; '.join(str(w) for w in warnings)}")
    return "\n".join(lines)
