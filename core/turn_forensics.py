"""Compact per-turn forensics persisted on assistant messages."""

from __future__ import annotations

from typing import Any, Mapping

from core.composer_attachments import ComposerAttachment, attachment_summary
from core.turn_context import TurnContext

FORENSICS_SCHEMA_VERSION = 1


def is_diagnostics_tool_id(tool_id: str) -> bool:
    from core.composer_attachments import canonical_composer_tool_id

    canon = canonical_composer_tool_id(tool_id or "")
    tid = canon.strip().lower()
    return tid == "diagnostics" or tid.startswith("diagnostics::")


def parse_diagnostics_tool_id(tool_id: str) -> tuple[str | None, int]:
    """
  Return optional target assistant message id and thread depth (pairs).
  ``diagnostics`` alone → latest assistant message.
  """
    raw = (tool_id or "").strip()
    lower = raw.lower()
    if lower == "diagnostics":
        return None, 0
    prefix = "diagnostics::"
    if not lower.startswith(prefix):
        return None, 0
    rest = raw[len(prefix) :].strip()
    if not rest:
        return None, 0
    if "::" in rest:
        msg_id, _, depth_s = rest.partition("::")
        try:
            depth = max(0, min(5, int(depth_s)))
        except ValueError:
            depth = 0
        return msg_id.strip() or None, depth
    return rest, 0


def format_diagnostics_tool_token(
    *,
    message_id: str | None = None,
    thread_depth: int = 0,
) -> str:
    if message_id:
        if thread_depth > 0:
            return f"diagnostics::{message_id}::{thread_depth}"
        return f"diagnostics::{message_id}"
    return "diagnostics"


def build_turn_forensics_snapshot(
    *,
    routing_turn_id: int | None,
    routing_record: Any | None = None,
    turn_context: TurnContext | None = None,
    degeneration: Any | None = None,
    composer_attachments: list[ComposerAttachment] | None = None,
    conversation_health: Any | None = None,
    effective_route: str = "",
    strategy: str = "",
) -> dict[str, Any]:
    routing_summary: dict[str, Any] | None = None
    if routing_record is not None:
        try:
            routing_summary = {
                "timestamp": float(getattr(routing_record, "timestamp", 0) or 0),
                "session_id": getattr(routing_record, "session_id", None),
                "turn_id": getattr(routing_record, "turn_id", None),
                "route": str(getattr(routing_record, "route", "") or ""),
                "route_pre_policy": getattr(routing_record, "route_pre_policy", None),
                "strategy": str(getattr(routing_record, "strategy", "") or ""),
                "top_intent": getattr(routing_record, "top_intent", None),
                "top_score": getattr(routing_record, "top_score", None),
                "summary": str(getattr(routing_record, "summary", "") or "")[:500],
            }
            trace = getattr(routing_record, "trace", None)
            if isinstance(trace, dict) and trace.get("retrieval_outcome"):
                routing_summary["retrieval_outcome"] = trace.get("retrieval_outcome")
        except Exception:
            routing_summary = None

    out: dict[str, Any] = {
        "schema_version": FORENSICS_SCHEMA_VERSION,
        "routing_turn_id": routing_turn_id,
        "effective_route": (effective_route or "").strip().lower(),
        "strategy": (strategy or "").strip(),
        "composer_attachments": _attachments_list(composer_attachments),
        "composer_attachment_summary": attachment_summary(
            list(composer_attachments or [])
        ),
    }
    if routing_summary:
        out["routing_summary"] = routing_summary
    elif (effective_route or strategy) and routing_turn_id is not None:
        out["routing_summary"] = {
            "turn_id": routing_turn_id,
            "route": effective_route,
            "strategy": strategy,
        }
    if turn_context is not None:
        try:
            out["turn_context"] = turn_context.trace_fields()
        except Exception:
            pass
    if degeneration is not None:
        try:
            out["degeneration"] = degeneration.trace_fields()
        except Exception:
            pass
    if conversation_health is not None:
        try:
            if hasattr(conversation_health, "trace_fields"):
                out["conversation_health"] = conversation_health.trace_fields()
            elif isinstance(conversation_health, Mapping):
                out["conversation_health"] = dict(conversation_health)
        except Exception:
            pass
    return out


def forensics_from_sources_payload(raw: Any) -> dict[str, Any] | None:
    from core.knowledge.ui_sources_payload import decode_forensics_payload

    return decode_forensics_payload(raw)


def _attachments_list(attachments: list[ComposerAttachment] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for att in attachments or []:
        rows.append({"kind": att.kind, "id": att.id, "label": att.label})
    return rows


def routing_record_from_forensics(forensics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not forensics:
        return None
    summary = forensics.get("routing_summary")
    return summary if isinstance(summary, dict) else None