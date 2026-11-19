"""Export helpers for turn diagnostics bundles and issue-report snippets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import user_data_root


def build_issue_report_snippet(bundle: dict[str, Any]) -> str:
    """Markdown block suitable for GitHub issues (no LLM required)."""
    orch = bundle.get("orchestration") or {}
    routing = orch.get("routing") or {}
    record = routing.get("record") if isinstance(routing.get("record"), dict) else {}
    inference = bundle.get("inference_snapshot") or {}
    native = inference.get("native") if isinstance(inference.get("native"), dict) else {}

    route = record.get("route") or routing.get("status") or "unknown"
    strategy = record.get("strategy") or orch.get("forensics", {}).get("strategy") or "—"
    model = (
        native.get("model_path_basename")
        or native.get("model_basename")
        or inference.get("engine_mode")
        or "—"
    )

    lines = [
        "### Qube issue report (diagnostics)",
        "",
        f"- **App version:** {bundle.get('app_version', '—')}",
        f"- **Session:** `{bundle.get('session_id_short', '—')}`",
        f"- **Assistant message:** `{bundle.get('target_assistant_message_id', '—')}`",
        f"- **Route:** {route}",
        f"- **Strategy:** {strategy}",
        f"- **Model / engine:** {model}",
        "",
        "**User question (excerpt):**",
        f"> {(bundle.get('target_turn') or {}).get('user_excerpt', '—')[:500]}",
        "",
        "**Assistant reply (excerpt):**",
        f"> {(bundle.get('target_turn') or {}).get('assistant_excerpt', '—')[:500]}",
        "",
        "**Routing note:**",
        f"> {(record.get('summary') or routing.get('hint') or '—')[:400]}",
    ]
    return "\n".join(lines)


def diagnostics_exports_dir() -> Path:
    path = user_data_root() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_bundle_json(bundle: dict[str, Any], *, message_id: str | None = None) -> Path:
    """Write bundle JSON under ~/.qube/exports/ and return the path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mid = (message_id or bundle.get("target_assistant_message_id") or "turn")[:16]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(mid))
    out = diagnostics_exports_dir() / f"turn_diagnostics_{safe}_{stamp}.json"
    out.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out


def bundle_json_text(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, ensure_ascii=False, default=str)