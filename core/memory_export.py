"""
Export LanceDB memory rows to human-readable Markdown.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable


def export_memories_to_markdown(rows: Iterable[dict], *, title: str = "Qube Memory Export") -> str:
    """Render memory rows (id, source, payload dict) into Markdown sections by tier."""
    from core.memory_retrieval_policy import tier_from_source

    buckets: dict[str, list[tuple[str, str, dict]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "")
        tier = tier_from_source(source)
        try:
            payload = row.get("payload")
            if payload is None:
                payload = json.loads(row.get("text", "{}") or "{}")
        except Exception:
            payload = {}
        content = (payload.get("content") or "").strip()
        if not content:
            continue
        rid = str(row.get("id") or "")
        buckets.setdefault(tier, []).append((rid, source, payload))

    lines = [
        f"# {title}",
        "",
        f"_Exported {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    for tier in ("preference", "knowledge", "episode", "context", "legacy"):
        items = buckets.get(tier) or []
        if not items:
            continue
        lines.append(f"## {tier.title()}")
        lines.append("")
        for rid, source, payload in items:
            lines.append(f"### {payload.get('category', 'fact')} — `{rid[:8]}`")
            lines.append("")
            lines.append(payload.get("content") or "")
            lines.append("")
            prov = (payload.get("provenance_quote") or "").strip()
            if prov:
                lines.append(f"> {prov}")
                lines.append("")
            msg_ids = payload.get("source_message_ids") or []
            if msg_ids:
                lines.append(f"_message ids: {', '.join(str(x) for x in msg_ids)}_")
                lines.append("")
            lines.append(f"_source: `{source}`_")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def default_export_path() -> str:
    home = os.path.expanduser("~")
    out_dir = os.path.join(home, ".qube", "exports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    return os.path.join(out_dir, f"memory_{stamp}.md")


def write_memory_export(rows: Iterable[dict], path: str | None = None) -> str:
    """Write export file; returns absolute path."""
    target = path or default_export_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    body = export_memories_to_markdown(rows)
    with open(target, "w", encoding="utf-8") as f:
        f.write(body)
    return target
