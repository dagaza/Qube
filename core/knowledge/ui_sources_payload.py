"""Encode/decode citation source payloads persisted on assistant messages."""

from __future__ import annotations

import json
from typing import Any

SOURCES_PAYLOAD_V2 = "qube_sources_v2"


def encode_sources_payload(
    sources: list[dict],
    *,
    transparency: dict[str, Any] | None = None,
) -> str | None:
    rows = [s for s in (sources or []) if isinstance(s, dict)]
    if not rows and not transparency:
        return None
    if transparency:
        return json.dumps(
            {
                "_format": SOURCES_PAYLOAD_V2,
                "sources": rows,
                "transparency": transparency,
            },
            ensure_ascii=False,
        )
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
