"""Formatters for session integration egress summary (Phase 3 / #61)."""

from __future__ import annotations

from typing import Sequence

from core.app_settings import get_advanced_engine_unlocked
from core.integrations.session_egress import IntegrationEgressRecord, session_egress_ledger

__all__ = [
    "include_raw_tools_in_egress",
    "format_egress_record_line",
    "format_session_egress_summary",
    "format_privacy_report_integrations_section",
    "get_session_egress_records",
]


def include_raw_tools_in_egress() -> bool:
    """When Advanced settings are unlocked, show raw provider tool ids."""
    return get_advanced_engine_unlocked()


def get_session_egress_records(
    session_id: str,
    *,
    include_raw_tools: bool | None = None,
) -> list[dict]:
    """Return serializable egress rows for a session."""
    show_raw = (
        include_raw_tools_in_egress()
        if include_raw_tools is None
        else include_raw_tools
    )
    rows: list[dict] = []
    for record in session_egress_ledger.records_for_session(session_id):
        payload = record.to_dict()
        if not show_raw:
            payload.pop("raw_tool", None)
        rows.append(payload)
    return rows


def format_egress_record_line(
    record: IntegrationEgressRecord | dict,
    *,
    include_raw_tools: bool | None = None,
) -> str:
    """Single-line summary for one integration call."""
    show_raw = (
        include_raw_tools_in_egress()
        if include_raw_tools is None
        else include_raw_tools
    )
    if isinstance(record, dict):
        provider = record.get("provider_id") or "?"
        server = record.get("server_id") or "?"
        group = record.get("capability_group") or "?"
        tier = record.get("tier") or "?"
        allowed = record.get("allowed")
        status = "ok" if allowed else "denied"
        line = f"{provider}/{server} · {group} ({tier}) — {status}"
        if show_raw and record.get("raw_tool"):
            line += f" · raw: {record['raw_tool']}"
        return line
    line = (
        f"{record.provider_id}/{record.server_id} · "
        f"{record.capability_group} ({record.tier}) — "
        f"{'ok' if record.allowed else 'denied'}"
    )
    if show_raw and record.raw_tool:
        line += f" · raw: {record.raw_tool}"
    return line


def format_session_egress_summary(
    session_id: str,
    *,
    include_raw_tools: bool | None = None,
    empty_message: str = "No integration calls this session.",
) -> str:
    """Multi-line human-readable egress summary for Telemetry / privacy UI."""
    records = session_egress_ledger.records_for_session(session_id)
    if not records:
        return empty_message
    lines = [f"Session integrations ({len(records)} call(s)):"]
    for index, record in enumerate(records, start=1):
        lines.append(f"  {index}. {format_egress_record_line(record, include_raw_tools=include_raw_tools)}")
    return "\n".join(lines)


def format_privacy_report_integrations_section(
    session_id: str,
    *,
    include_raw_tools: bool | None = None,
) -> str:
    """Section body for a one-click privacy report export."""
    header = "## Integrations (this session)\n"
    body = format_session_egress_summary(
        session_id,
        include_raw_tools=include_raw_tools,
        empty_message="No integration capability calls were recorded this session.",
    )
    return header + body + "\n"
