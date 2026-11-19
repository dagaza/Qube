"""Encode/decode citation source payloads persisted on assistant messages."""

from __future__ import annotations

import json
from typing import Any

SOURCES_PAYLOAD_V2 = "qube_sources_v2"


def encode_sources_payload(
    sources: list[dict],
    *,
    transparency: dict[str, Any] | None = None,
    forensics: dict[str, Any] | None = None,
) -> str | None:
    rows = [s for s in (sources or []) if isinstance(s, dict)]
    if not rows and not transparency and not forensics:
        return None
    if transparency or forensics:
        payload: dict[str, Any] = {
            "_format": SOURCES_PAYLOAD_V2,
            "sources": rows,
        }
        if transparency:
            payload["transparency"] = transparency
        if forensics:
            payload["forensics"] = forensics
        return json.dumps(payload, ensure_ascii=False)
    return json.dumps(rows, ensure_ascii=False)


def decode_sources_payload(raw: Any) -> tuple[list[dict], dict[str, Any] | None]:
    if raw is None:
        return [], None
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)], None
    if isinstance(raw, str):
        if not raw.strip():
            return [], None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [], None
        return decode_sources_payload(parsed)
    if isinstance(raw, dict) and raw.get("_format") == SOURCES_PAYLOAD_V2:
        sources = [s for s in (raw.get("sources") or []) if isinstance(s, dict)]
        transparency = raw.get("transparency")
        if isinstance(transparency, dict) and transparency:
            return sources, transparency
        return sources, None
    return [], None


def decode_forensics_payload(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    parsed = raw
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict) and parsed.get("_format") == SOURCES_PAYLOAD_V2:
        forensics = parsed.get("forensics")
        return forensics if isinstance(forensics, dict) and forensics else None
    return None
