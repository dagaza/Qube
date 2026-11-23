"""Read retrieval traces from the web search audit JSONL log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.knowledge.observability import RETRIEVAL_TRACE_EVENT
from core.knowledge.search_outcome import (
    format_search_outcome_summary_line,
    search_outcome_from_relevance_diag,
)
from core.web_search_audit_sink import default_web_search_audit_log_path


def _parse_line(line: str) -> dict[str, Any] | None:
    line = (line or "").strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("event") != RETRIEVAL_TRACE_EVENT:
        return None
    return payload


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

    for line in reversed(lines):
        payload = _parse_line(line)
        if payload is None:
            continue
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
    warnings = trace.get("warnings") or []
    if warnings:
        lines.append(f"Warnings: {'; '.join(str(w) for w in warnings)}")
    return "\n".join(lines)
