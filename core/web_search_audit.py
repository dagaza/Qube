"""Structured JSONL audit for web search attempts (observer-only, opt-in)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from core.web_search_audit_sink import WEB_SEARCH_AUDIT_LOGGER_NAME

WEB_SEARCH_AUDIT_SCHEMA_VERSION = 1

PROVIDER = "duckduckgo_html"
PROVIDER_ENDPOINT = "https://html.duckduckgo.com/html/"
AUDIT_NOTE = "SERP snippets only; result pages are not fetched."

MAX_QUERY_CHARS = 500
MAX_TITLE_CHARS = 120
MAX_SNIPPET_CHARS = 240

STATUS_VETOED_TOOL_DISABLED = "vetoed_tool_disabled"
STATUS_VETOED_UNGROUNDED = "vetoed_ungrounded"
STATUS_SUCCESS = "success"
STATUS_NO_RESULTS = "no_results"
STATUS_NETWORK_ERROR = "network_error"
STATUS_RELEVANCE_FILTERED = "relevance_filtered"
STATUS_EMPTY_SENTINEL = "empty_sentinel"


def web_search_audit_log_env_override() -> bool | None:
    raw = os.getenv("QUBE_WEB_SEARCH_AUDIT_LOG")
    if raw is None:
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def web_search_audit_redact_enabled() -> bool:
    return str(os.getenv("QUBE_WEB_SEARCH_AUDIT_REDACT", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def web_search_audit_log_enabled() -> bool:
    """True when web search audit JSONL recording is active."""
    override = web_search_audit_log_env_override()
    if override is not None:
        return override
    try:
        from core.app_settings import get_web_search_audit_log_enabled

        return get_web_search_audit_log_enabled()
    except Exception:
        return False


@dataclass(frozen=True)
class WebSearchResultAuditRow:
    rank: int
    title: str
    url: str | None
    snippet_preview: str
    kept: bool
    token_overlap: float | None = None
    semantic_score: float | None = None


@dataclass(frozen=True)
class WebSearchAuditEvent:
    schema_version: int
    ts: float
    request_id: str
    session_id: str | None
    turn_id: int | None
    trigger: str
    execution_route: str
    internet_tool_enabled: bool
    user_prompt: str
    query_raw: str
    query_resolved: str
    query_rewrite_reason: str | None
    query_rewrite_failed: bool
    target_site: str | None
    provider: str
    provider_endpoint: str
    status: str
    latency_ms: float | None
    results_raw_count: int
    results_kept_count: int
    relevance_dropped_count: int
    relevance_min_overlap: float | None
    results: tuple[WebSearchResultAuditRow, ...]
    query_redacted: bool = False


def resolve_web_search_trigger(
    *,
    force_web: bool,
    manual_web: bool,
    composer_internet: bool,
    auto_web: bool,
    execution_route: str,
) -> str:
    if composer_internet:
        return "composer_internet"
    if force_web:
        return "force_toggle"
    if manual_web:
        return "manual_phrase"
    if auto_web:
        return "auto_web"
    route = str(execution_route or "").upper()
    if route in ("WEB", "INTERNET"):
        return f"router_{route.lower()}"
    if route == "HYBRID":
        return "router_hybrid"
    return "unknown"


def infer_web_search_status(
    *,
    veto_status: str | None,
    web_results_raw: Sequence[Mapping[str, Any]] | None,
    web_results_kept: Sequence[Mapping[str, Any]] | None,
    relevance_diag: Mapping[str, Any] | None,
) -> str:
    if veto_status:
        return veto_status

    raw_items = [r for r in (web_results_raw or []) if isinstance(r, Mapping)]
    if not raw_items:
        return STATUS_NO_RESULTS

    snippets = " ".join(str(r.get("snippet") or "") for r in raw_items)
    if "Internet search failed" in snippets:
        return STATUS_NETWORK_ERROR
    if "No relevant internet results found" in snippets or not snippets.strip():
        return STATUS_EMPTY_SENTINEL

    kept_count = 0
    if relevance_diag is not None:
        try:
            kept_count = int(relevance_diag.get("web_results_kept_count") or 0)
        except (TypeError, ValueError):
            kept_count = 0
    elif web_results_kept is not None:
        kept_count = len([r for r in web_results_kept if isinstance(r, Mapping)])

    if kept_count <= 0:
        return STATUS_RELEVANCE_FILTERED
    return STATUS_SUCCESS


def _truncate(text: str, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _redact_query(text: str) -> str:
    digest = hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"[redacted sha256:{digest}]"


def _same_result(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    au = str(a.get("url") or "").strip()
    bu = str(b.get("url") or "").strip()
    if au and bu and au == bu:
        return True
    return str(a.get("title") or "").strip() == str(b.get("title") or "").strip()


def _dropped_lookup(
    dropped: Sequence[Mapping[str, Any]] | None,
    title: str,
) -> Mapping[str, Any] | None:
    title_key = str(title or "").strip()
    short = _truncate(title_key, 80)
    for row in dropped or []:
        if not isinstance(row, Mapping):
            continue
        row_title = str(row.get("title") or "").strip()
        if row_title == title_key or row_title == short:
            return row
    return None


def build_result_audit_rows(
    web_results_raw: Sequence[Mapping[str, Any]] | None,
    web_results_kept: Sequence[Mapping[str, Any]] | None,
    relevance_diag: Mapping[str, Any] | None,
    *,
    redact_snippets: bool,
) -> tuple[WebSearchResultAuditRow, ...]:
    raw_items = [dict(r) for r in (web_results_raw or []) if isinstance(r, Mapping)]
    kept_items = [dict(r) for r in (web_results_kept or []) if isinstance(r, Mapping)]
    dropped = relevance_diag.get("web_relevance_dropped") if relevance_diag else None
    dropped_list = [d for d in (dropped or []) if isinstance(d, Mapping)]

    rows: list[WebSearchResultAuditRow] = []
    for rank, item in enumerate(raw_items, start=1):
        title = _truncate(str(item.get("title") or "").strip(), MAX_TITLE_CHARS)
        url = str(item.get("url") or "").strip() or None
        if url and not url.startswith(("http://", "https://")):
            url = None

        kept_flag = any(_same_result(item, kept) for kept in kept_items)
        drop_info = _dropped_lookup(dropped_list, title)

        token_overlap = item.get("_web_token_overlap")
        semantic_score = item.get("_web_semantic_score")
        if drop_info is not None:
            token_overlap = drop_info.get("token_overlap", token_overlap)
            semantic_score = drop_info.get("semantic_score", semantic_score)
        elif kept_flag:
            for kept in kept_items:
                if _same_result(item, kept):
                    token_overlap = kept.get("_web_token_overlap", token_overlap)
                    semantic_score = kept.get("_web_semantic_score", semantic_score)
                    break

        snippet = _truncate(str(item.get("snippet") or "").strip(), MAX_SNIPPET_CHARS)
        if redact_snippets:
            snippet = ""

        rows.append(
            WebSearchResultAuditRow(
                rank=rank,
                title=title,
                url=url,
                snippet_preview=snippet,
                kept=kept_flag,
                token_overlap=_coerce_float(token_overlap),
                semantic_score=_coerce_float(semantic_score),
            )
        )
    return tuple(rows)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_audit_event_from_llm_turn(
    *,
    session_id: str | None,
    turn_id: int | None,
    user_prompt: str,
    execution_route: str,
    internet_tool_enabled: bool,
    force_web: bool,
    manual_web: bool,
    auto_web: bool,
    composer_internet: bool,
    query_raw: str,
    query_resolved: str,
    query_rewrite_reason: str | None,
    query_rewrite_failed: bool,
    target_site: str | None = None,
    veto_status: str | None = None,
    web_results_raw: Sequence[Mapping[str, Any]] | None = None,
    web_results_kept: Sequence[Mapping[str, Any]] | None = None,
    relevance_diag: Mapping[str, Any] | None = None,
    latency_ms: float | None = None,
    request_id: str | None = None,
    ts: float | None = None,
) -> WebSearchAuditEvent:
    redact = web_search_audit_redact_enabled()
    status = infer_web_search_status(
        veto_status=veto_status,
        web_results_raw=web_results_raw,
        web_results_kept=web_results_kept,
        relevance_diag=relevance_diag,
    )
    result_rows = build_result_audit_rows(
        web_results_raw,
        web_results_kept,
        relevance_diag,
        redact_snippets=redact,
    )
    raw_count = len([r for r in (web_results_raw or []) if isinstance(r, Mapping)])
    kept_count = len([r for r in result_rows if r.kept])
    dropped_count = len(relevance_diag.get("web_relevance_dropped") or []) if relevance_diag else 0

    prompt_out = _redact_query(user_prompt) if redact else _truncate(user_prompt, MAX_QUERY_CHARS)
    raw_out = _redact_query(query_raw) if redact else _truncate(query_raw, MAX_QUERY_CHARS)
    resolved_out = (
        _redact_query(query_resolved) if redact else _truncate(query_resolved, MAX_QUERY_CHARS)
    )

    return WebSearchAuditEvent(
        schema_version=WEB_SEARCH_AUDIT_SCHEMA_VERSION,
        ts=float(ts if ts is not None else time.time()),
        request_id=str(request_id or uuid.uuid4()),
        session_id=session_id,
        turn_id=turn_id,
        trigger=resolve_web_search_trigger(
            force_web=force_web,
            manual_web=manual_web,
            composer_internet=composer_internet,
            auto_web=auto_web,
            execution_route=execution_route,
        ),
        execution_route=str(execution_route or "NONE").upper(),
        internet_tool_enabled=bool(internet_tool_enabled),
        user_prompt=prompt_out,
        query_raw=raw_out,
        query_resolved=resolved_out,
        query_rewrite_reason=query_rewrite_reason,
        query_rewrite_failed=bool(query_rewrite_failed),
        target_site=target_site,
        provider=PROVIDER,
        provider_endpoint=PROVIDER_ENDPOINT,
        status=status,
        latency_ms=latency_ms,
        results_raw_count=raw_count,
        results_kept_count=kept_count,
        relevance_dropped_count=dropped_count,
        relevance_min_overlap=_coerce_float(
            relevance_diag.get("web_relevance_min_overlap") if relevance_diag else None
        ),
        results=result_rows,
        query_redacted=redact,
    )


def build_standalone_audit_event(
    *,
    query: str,
    raw_results: Sequence[Mapping[str, Any]] | None,
    error: str | None = None,
    latency_ms: float | None = None,
) -> WebSearchAuditEvent:
    status = STATUS_NETWORK_ERROR if error else infer_web_search_status(
        veto_status=None,
        web_results_raw=raw_results,
        web_results_kept=raw_results,
        relevance_diag=None,
    )
    return build_audit_event_from_llm_turn(
        session_id=None,
        turn_id=None,
        user_prompt=query,
        execution_route="NONE",
        internet_tool_enabled=True,
        force_web=False,
        manual_web=False,
        auto_web=False,
        composer_internet=False,
        query_raw=query,
        query_resolved=query,
        query_rewrite_reason=None,
        query_rewrite_failed=False,
        veto_status=status if error else None,
        web_results_raw=raw_results,
        web_results_kept=raw_results if status == STATUS_SUCCESS else None,
        relevance_diag=None,
        latency_ms=latency_ms,
    )


def serialize_audit_event(event: WebSearchAuditEvent) -> dict[str, Any]:
    return {
        "schema_version": event.schema_version,
        "event": "web_search_audit",
        "ts": event.ts,
        "request_id": event.request_id,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "trigger": event.trigger,
        "execution_route": event.execution_route,
        "internet_tool_enabled": event.internet_tool_enabled,
        "user_prompt": event.user_prompt,
        "query_raw": event.query_raw,
        "query_resolved": event.query_resolved,
        "query_rewrite_reason": event.query_rewrite_reason,
        "query_rewrite_failed": event.query_rewrite_failed,
        "target_site": event.target_site,
        "provider": event.provider,
        "provider_endpoint": event.provider_endpoint,
        "note": AUDIT_NOTE,
        "status": event.status,
        "latency_ms": event.latency_ms,
        "results_raw_count": event.results_raw_count,
        "results_kept_count": event.results_kept_count,
        "relevance_dropped_count": event.relevance_dropped_count,
        "relevance_min_overlap": event.relevance_min_overlap,
        "query_redacted": event.query_redacted,
        "results": [
            {
                "rank": row.rank,
                "title": row.title,
                "url": row.url,
                "snippet_preview": row.snippet_preview,
                "kept": row.kept,
                "token_overlap": row.token_overlap,
                "semantic_score": row.semantic_score,
            }
            for row in event.results
        ],
    }


def record_web_search_audit(event: WebSearchAuditEvent) -> None:
    """Append one JSONL line. No-op when disabled. Never raises."""
    if not web_search_audit_log_enabled():
        return
    try:
        payload = serialize_audit_event(event)
        logging.getLogger(WEB_SEARCH_AUDIT_LOGGER_NAME).info(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
    except Exception:
        pass
