"""Build Turn Diagnostics Bundles for composer @diagnostics turns."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.__version__ import __version__
from core.composer_attachments import attachment_summary
from core.database import DatabaseManager
from core.inference_transparency import aggregate_app_transparency
from core.knowledge.routing_inspect_explain import format_routing_inspect_text
from core.knowledge.ui_sources_payload import (
    decode_forensics_payload,
    decode_sources_payload,
)
from core.routing_debug_sink import default_routing_debug_log_path
from core.turn_forensics import routing_record_from_forensics

TDB_SCHEMA_VERSION = 1
_MAX_EXCERPT = 4000
_MAX_LOG_EXCERPTS = 8000
_THREAD_PAIR_DEPTH_DEFAULT = 3

_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._\-]{8,}\b")
_API_KEY_RE = re.compile(r"(?i)(api[_-]?key|token)\s*[:=]\s*\S+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _redact_text(text: str, *, max_len: int) -> str:
    s = str(text or "")
    s = _BEARER_RE.sub("[redacted bearer]", s)
    s = _API_KEY_RE.sub("[redacted credential]", s)
    if len(s) > max_len:
        return s[: max_len - 20] + " … [truncated]"
    return s


def _latest_assistant_message_id(db: DatabaseManager, session_id: str) -> str | None:
    rows = db.list_session_messages_raw(session_id)
    for row in reversed(rows):
        if str(row.get("role") or "").lower() == "assistant":
            mid = str(row.get("id") or "").strip()
            if mid:
                return mid
    return None


def _message_row(
    rows: list[dict], message_id: str
) -> tuple[dict | None, dict | None]:
    """Return (assistant_row, preceding_user_row) for ``message_id``."""
    idx: int | None = None
    for i, row in enumerate(rows):
        if str(row.get("id") or "") == message_id:
            idx = i
            break
    if idx is None:
        return None, None
    assistant = rows[idx]
    user_row = None
    for j in range(idx - 1, -1, -1):
        if str(rows[j].get("role") or "").lower() == "user":
            user_row = rows[j]
            break
    return assistant, user_row


def _thread_excerpt(
    rows: list[dict],
    *,
    through_index: int,
    pair_depth: int,
) -> list[dict[str, str]]:
    if pair_depth <= 0:
        return []
    excerpt: list[dict[str, str]] = []
    pairs = 0
    i = through_index
    while i >= 0 and pairs < pair_depth:
        if str(rows[i].get("role") or "").lower() != "assistant":
            i -= 1
            continue
        assistant = rows[i]
        user = None
        for j in range(i - 1, -1, -1):
            if str(rows[j].get("role") or "").lower() == "user":
                user = rows[j]
                break
        if user:
            excerpt.insert(
                0,
                {
                    "user": _redact_text(str(user.get("content") or ""), max_len=1200),
                    "assistant": _redact_text(
                        str(assistant.get("content") or ""), max_len=1200
                    ),
                    "assistant_message_id": str(assistant.get("id") or ""),
                },
            )
            pairs += 1
        i -= 1
    return excerpt


def _sources_summary(sources: list[dict]) -> dict[str, Any]:
    titles: list[str] = []
    types: dict[str, int] = {}
    for src in sources[:12]:
        if not isinstance(src, dict):
            continue
        st = str(src.get("type") or "unknown").lower()
        types[st] = types.get(st, 0) + 1
        title = str(src.get("filename") or src.get("title") or src.get("url") or "")[:120]
        if title:
            titles.append(title)
    return {"count": len(sources), "types": types, "sample_titles": titles}


def _sources_citation_catalog(sources: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in sources[:20]:
        if not isinstance(src, dict):
            continue
        rows.append(
            {
                "id": src.get("id"),
                "type": src.get("type"),
                "title": str(
                    src.get("filename") or src.get("title") or ""
                )[:160],
                "url": str(src.get("url") or src.get("link") or "")[:300],
            }
        )
    return rows


def _citation_integrity_block(
    assistant_content: str, sources: list[dict]
) -> dict[str, Any]:
    from core.citation_integrity import analyze_citations

    report = analyze_citations(assistant_content or "", sources)
    verdict = "no_sources_attached"
    if report.source_count > 0 and not report.cited_ids:
        verdict = "sources_attached_no_bracket_tokens_in_stored_text"
    elif report.orphan_ids:
        verdict = "orphan_bracket_tokens"
    elif report.cited_ids and not report.has_violation:
        verdict = "bracket_tokens_match_sources"
    elif report.missing_citation:
        verdict = "sources_present_but_uncited"
    elif report.source_count > 0:
        verdict = "sources_attached"

    return {
        "verdict": verdict,
        "verdict_note": _CITATION_VERDICT_NOTES.get(verdict, ""),
        "cited_ids_in_stored_text": list(report.cited_ids),
        "valid_source_ids": list(report.valid_ids),
        "orphan_ids": list(report.orphan_ids),
        "source_count": report.source_count,
        "web_hit_count": report.web_hit_count,
        "integrity_violation": report.has_violation,
        "missing_citation_when_sources_present": report.missing_citation,
    }


_CITATION_VERDICT_NOTES: dict[str, str] = {
    "bracket_tokens_match_sources": (
        "Bracket markers in the stored assistant message match persisted source ids. "
        "The Sources panel and inline links use the same sources_json payload."
    ),
    "sources_attached_no_bracket_tokens_in_stored_text": (
        "Sources were saved on the message but the stored assistant text has no [n] "
        "tokens (common after citation repair or for some render paths). That does "
        "not mean citations are hallucinated — check sources_citation_catalog."
    ),
    "orphan_bracket_tokens": (
        "Stored text cites ids that are not in sources_json (true integrity issue)."
    ),
    "sources_present_but_uncited": (
        "Sources were attached but the stored answer body cites none of them."
    ),
    "no_sources_attached": "No sources_json rows on this message.",
    "sources_attached": "Sources attached; see cited_ids_in_stored_text.",
}


def _logging_state_block(routing: dict[str, Any]) -> dict[str, Any]:
    from core.app_settings import (
        get_diagnostics_technical_detail_enabled,
        get_llm_debug_log_file_enabled,
        get_routing_debug_log_enabled,
    )

    routing_status = str(routing.get("status") or "")
    return {
        "routing_debug_log_enabled": get_routing_debug_log_enabled(),
        "llm_debug_log_file_enabled": get_llm_debug_log_file_enabled(),
        "diagnostics_technical_detail_enabled": get_diagnostics_technical_detail_enabled(),
        "routing_capture_for_turn": routing_status,
        "routing_capture_source": routing.get("source"),
        "notes": [
            "routing_debug.log stores per-turn routing JSONL (route, strategy, outcomes). "
            "It does not store the full model prompt unless other debug tooling is on.",
            "Enable diagnostics **technical detail** (Privacy & data) to attach llm_debug "
            "excerpts to @diagnostics bundles — not the routing log toggle.",
            "If routing_capture_for_turn is 'recorded', do not tell the user to enable "
            "routing debug logging for this turn.",
        ],
    }


def _composer_on_user_turn(user_content: str) -> dict[str, Any]:
    from core.composer_attachments import parse_attachments

    clean, attachments = parse_attachments(user_content or "")
    if not attachments:
        return {"detected": False, "attachments": []}
    return {
        "detected": True,
        "clean_user_prompt": clean[:500],
        "attachments": [
            {"kind": a.kind, "id": a.id, "label": a.label} for a in attachments
        ],
    }


def _compact_routing_record(record: Any) -> dict[str, Any]:
    if record is None:
        return {}
    if dataclasses.is_dataclass(record):
        base = {
            "timestamp": float(getattr(record, "timestamp", 0) or 0),
            "session_id": getattr(record, "session_id", None),
            "turn_id": getattr(record, "turn_id", None),
            "route": getattr(record, "route", None),
            "route_pre_policy": getattr(record, "route_pre_policy", None),
            "strategy": getattr(record, "strategy", None),
            "top_intent": getattr(record, "top_intent", None),
            "top_score": getattr(record, "top_score", None),
            "summary": str(getattr(record, "summary", "") or "")[:800],
        }
        trace = getattr(record, "trace", None)
        if isinstance(trace, dict):
            if trace.get("retrieval_outcome"):
                base["retrieval_outcome"] = trace.get("retrieval_outcome")
        return base
    if isinstance(record, Mapping):
        return dict(record)
    return {}


def _find_routing_in_buffer(
    buffer: Any,
    *,
    session_id: str,
    turn_id: int | None,
) -> dict[str, Any] | None:
    if buffer is None or turn_id is None:
        return None
    try:
        snaps = buffer.snapshot()
    except Exception:
        return None
    for rec in reversed(snaps):
        if getattr(rec, "session_id", None) != session_id:
            continue
        if int(getattr(rec, "turn_id", -1) or -1) != int(turn_id):
            continue
        return _compact_routing_record(rec)
    return None


def _find_routing_in_jsonl(
    *,
    session_id: str,
    turn_id: int | None,
    max_lines: int = 400,
) -> dict[str, Any] | None:
    if turn_id is None:
        return None
    path = default_routing_debug_log_path()
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-max_lines:]):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(payload.get("session_id") or "") != session_id:
            continue
        if payload.get("turn_id") != turn_id:
            continue
        return {
            "timestamp": payload.get("ts"),
            "session_id": payload.get("session_id"),
            "turn_id": payload.get("turn_id"),
            "route": payload.get("route"),
            "route_pre_policy": payload.get("route_pre_policy"),
            "strategy": payload.get("strategy"),
            "top_intent": payload.get("top_intent"),
            "top_score": payload.get("top_score"),
            "summary": payload.get("summary"),
            "retrieval_outcome": payload.get("retrieval_outcome"),
        }
    return None


def _routing_block(
    *,
    forensics: dict[str, Any] | None,
    buffer: Any,
    session_id: str,
    turn_id: int | None,
) -> dict[str, Any]:
    stored = routing_record_from_forensics(forensics)
    if stored:
        return {"status": "recorded", "source": "message_forensics", "record": stored}
    live = _find_routing_in_buffer(buffer, session_id=session_id, turn_id=turn_id)
    if live:
        return {"status": "recorded", "source": "routing_debug_buffer", "record": live}
    logged = _find_routing_in_jsonl(session_id=session_id, turn_id=turn_id)
    if logged:
        return {"status": "recorded", "source": "routing_debug_log", "record": logged}
    return {
        "status": "not_recorded",
        "hint": (
            "Enable **Routing debug log** under Settings → Privacy & data to "
            "capture per-turn routing for future replies."
        ),
    }


def _log_excerpts_for_turn(
    *,
    session_id: str,
    turn_id: int | None,
    technical: bool,
) -> dict[str, str]:
    from core.diagnostic_logs import DIAGNOSTIC_LOGS, read_log_tail

    out: dict[str, str] = {}
    budget = _MAX_LOG_EXCERPTS
    specs = [s for s in DIAGNOSTIC_LOGS if s.id == "routing_debug"]
    if technical:
        specs.extend(s for s in DIAGNOSTIC_LOGS if s.id == "llm_debug")
    for spec in specs:
        if budget <= 0:
            break
        path = spec.path_fn()
        tail = read_log_tail(path, max_lines=80 if spec.id == "routing_debug" else 40)
        if turn_id is not None and session_id:
            filtered = []
            for line in tail.splitlines():
                if str(turn_id) in line and session_id[:8] in line:
                    filtered.append(line)
                elif spec.id == "routing_debug" and f'"turn_id": {turn_id}' in line:
                    filtered.append(line)
            if filtered:
                tail = "\n".join(filtered[-40:])
        piece = tail[-min(len(tail), budget) :]
        out[spec.id] = piece
        budget -= len(piece)
    return out


def build_turn_diagnostics_bundle(
    db: DatabaseManager,
    *,
    session_id: str,
    target_assistant_message_id: str | None = None,
    thread_depth: int = 0,
    routing_debug_buffer: Any = None,
    native_engine: Any = None,
    embedder: Any = None,
    technical_detail: bool = False,
) -> dict[str, Any]:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")

    rows = db.list_session_messages_raw(sid)
    target_id = (target_assistant_message_id or "").strip() or _latest_assistant_message_id(
        db, sid
    )
    if not target_id:
        return {
            "schema_version": TDB_SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "app_version": __version__,
            "session_id": sid,
            "error": "no_assistant_message",
            "hint": "Send a message and get a reply before using @diagnostics.",
        }

    assistant_row, user_row = _message_row(rows, target_id)
    if assistant_row is None:
        return {
            "schema_version": TDB_SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "app_version": __version__,
            "session_id": sid,
            "target_assistant_message_id": target_id,
            "error": "message_not_found",
        }

    depth = thread_depth if thread_depth > 0 else _THREAD_PAIR_DEPTH_DEFAULT
    idx = next(
        (i for i, r in enumerate(rows) if str(r.get("id") or "") == target_id),
        len(rows) - 1,
    )

    sources, transparency = decode_sources_payload(assistant_row.get("sources_json"))
    forensics = decode_forensics_payload(assistant_row.get("sources_json"))
    turn_id = None
    if forensics:
        turn_id = forensics.get("routing_turn_id")
    if turn_id is None:
        rec = db.get_retrieval_record(bundle_id=str(assistant_row.get("evidence_bundle_id") or ""))
        if rec:
            turn_id = rec.get("turn_id")

    routing = _routing_block(
        forensics=forensics,
        buffer=routing_debug_buffer,
        session_id=sid,
        turn_id=int(turn_id) if turn_id is not None else None,
    )
    routing_record = routing.get("record") if isinstance(routing.get("record"), dict) else None

    retrieval_record = None
    bundle_id = str(assistant_row.get("evidence_bundle_id") or "").strip()
    if bundle_id:
        retrieval_record = db.get_retrieval_record(bundle_id=bundle_id)

    inspect_summary = ""
    if routing_record or retrieval_record:
        try:
            inspect_summary = format_routing_inspect_text(
                routing_record,
                retrieval_outcome=(
                    (routing_record or {}).get("retrieval_outcome")
                    if routing_record
                    else None
                ),
            )
        except Exception:
            inspect_summary = ""

    inference = {}
    try:
        from core.app_settings import get_engine_mode

        inference = aggregate_app_transparency(
            native_engine=native_engine,
            embedder=embedder,
        )
        inference["engine_mode"] = get_engine_mode()
    except Exception:
        inference = {}

    assistant_full = str(assistant_row.get("content") or "")
    user_full = str(user_row.get("content") or "") if user_row else ""
    citations = _citation_integrity_block(assistant_full, sources)
    logging_state = _logging_state_block(routing)
    orchestration_extras: dict[str, Any] = {}
    if isinstance(transparency, dict):
        if transparency.get("knowledge_service"):
            orchestration_extras["knowledge_service"] = transparency.get(
                "knowledge_service"
            )
        if transparency.get("retrieval_strategy"):
            orchestration_extras["retrieval_strategy"] = transparency.get(
                "retrieval_strategy"
            )

    bundle: dict[str, Any] = {
        "schema_version": TDB_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "app_version": __version__,
        "session_id": sid,
        "session_id_short": sid[:8] + "…" if len(sid) > 8 else sid,
        "target_assistant_message_id": target_id,
        "target_user_message_id": str(user_row.get("id") or "") if user_row else None,
        "routing_turn_id": turn_id,
        "privacy_tier": "technical" if technical_detail else "standard",
        "target_turn": {
            "user_excerpt": _redact_text(
                str(user_row.get("content") or "") if user_row else "",
                max_len=_MAX_EXCERPT,
            ),
            "assistant_excerpt": _redact_text(
                str(assistant_row.get("content") or ""),
                max_len=_MAX_EXCERPT,
            ),
        },
        "thread_excerpt": _thread_excerpt(rows, through_index=idx, pair_depth=depth),
        "orchestration": {
            "routing": routing,
            "forensics": forensics,
            "composer_attachment_summary": (
                (forensics or {}).get("composer_attachment_summary") or ""
            ),
            "composer_on_analyzed_turn": _composer_on_user_turn(user_full),
            **orchestration_extras,
        },
        "retrieval": {
            "evidence_bundle_id": bundle_id or None,
            "retrieval_record": retrieval_record,
            "sources_summary": _sources_summary(sources),
            "sources_citation_catalog": _sources_citation_catalog(sources),
            "evidence_transparency": transparency,
            "inspect_summary": inspect_summary[:6000] if inspect_summary else "",
        },
        "citations": citations,
        "logging_state": logging_state,
        "inference_snapshot": inference,
        "log_excerpts": _log_excerpts_for_turn(
            session_id=sid,
            turn_id=int(turn_id) if turn_id is not None else None,
            technical=technical_detail,
        ),
    }
    return bundle