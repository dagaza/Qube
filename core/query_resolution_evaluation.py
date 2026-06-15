"""
Offline evaluation for discourse query resolution and fixture-based web retrieval.

Pure logic (no Qt). Used by ``tools/evaluate_query_resolution.py`` and
``core/router_evaluation`` when cases carry query-resolution expectations.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from core.discourse_query import ResolvedRetrievalQuery, build_resolved_retrieval_query
from core.retrieval_relevance import filter_web_results
from mcp.internet_tool import parse_ddg_html_results

logger = logging.getLogger("Qube.QueryResolutionEval")

QUERY_RESOLUTION_CORPUS_SCHEMA = "qube.query_resolution_corpus.v1"
DEFAULT_WEB_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "web"


@dataclass(frozen=True)
class QueryResolutionExpectations:
    """Substring and retrieval expectations for a single eval case."""

    inference_contains: tuple[str, ...] = ()
    inference_not_contains: tuple[str, ...] = ()
    web_contains: tuple[str, ...] = ()
    web_not_contains: tuple[str, ...] = ()
    routing_contains: tuple[str, ...] = ()
    retrieval_contains: tuple[str, ...] = ()
    min_web_hits: int = 0
    web_fixture_id: str = ""


@dataclass(frozen=True)
class QueryResolutionEvalCase:
    id: str
    prompt: str
    category: str
    notes: str = ""
    history: tuple[dict[str, str], ...] = ()
    expect: QueryResolutionExpectations = field(
        default_factory=QueryResolutionExpectations
    )
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResolutionEvalResult:
    case_id: str
    prompt: str
    category: str
    notes: str
    raw_text: str
    inference_text: str
    routing_text: str
    retrieval_text: str
    web_text: str
    web_rewrite_reason: str
    inference_rewrite_applied: bool
    resolution_pass: bool
    failed_checks: list[str] = field(default_factory=list)
    web_fixture_id: str = ""
    web_fixture_raw_count: int = 0
    web_fixture_hits: int = 0
    web_relevance_gate_dropped: int = 0
    error: str = ""


@dataclass
class QueryResolutionEvalSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float
    web_fixture_cases: int
    web_fixture_hit_rate: float
    by_category: dict[str, dict[str, Any]]
    failures: list[dict[str, Any]]


def _parse_history(raw: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, list):
        return ()
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if content:
            out.append({"role": role, "content": content})
    return tuple(out)


def _parse_expectations(raw: Any) -> QueryResolutionExpectations:
    if not isinstance(raw, dict):
        return QueryResolutionExpectations()

    def _strings(key: str) -> tuple[str, ...]:
        val = raw.get(key)
        if not isinstance(val, list):
            return ()
        return tuple(str(x).strip() for x in val if str(x).strip())

    fixture_id = str(raw.get("web_fixture_id") or "").strip()
    min_hits = raw.get("min_web_hits", 0)
    try:
        min_web_hits = max(0, int(min_hits))
    except (TypeError, ValueError):
        min_web_hits = 0

    return QueryResolutionExpectations(
        inference_contains=_strings("inference_contains"),
        inference_not_contains=_strings("inference_not_contains"),
        web_contains=_strings("web_contains"),
        web_not_contains=_strings("web_not_contains"),
        routing_contains=_strings("routing_contains"),
        retrieval_contains=_strings("retrieval_contains"),
        min_web_hits=min_web_hits,
        web_fixture_id=fixture_id,
    )


def expectations_from_flags(flags: dict[str, Any] | None) -> QueryResolutionExpectations | None:
    """Parse optional ``flags.query_resolution`` block from router corpus cases."""
    if not isinstance(flags, dict):
        return None
    block = flags.get("query_resolution")
    if block is None:
        return None
    return _parse_expectations(block)


def load_query_resolution_corpus(path: Path) -> tuple[dict[str, Any], list[QueryResolutionEvalCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("corpus root must be a JSON object")

    schema = str(data.get("schema") or "")
    if schema and schema != QUERY_RESOLUTION_CORPUS_SCHEMA:
        raise ValueError(
            f"unsupported corpus schema: {schema!r} "
            f"(expected {QUERY_RESOLUTION_CORPUS_SCHEMA})"
        )

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("corpus must contain a non-empty 'cases' list")

    cases: list[QueryResolutionEvalCase] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{idx}] must be an object")
        case_id = str(raw.get("id") or f"qr_{idx:03d}").strip()
        if not case_id:
            raise ValueError(f"cases[{idx}] missing id")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)

        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"cases[{idx}] ({case_id}) missing prompt")

        category = str(raw.get("category") or "uncategorized").strip()
        notes = str(raw.get("notes") or "").strip()
        history = _parse_history(raw.get("history"))
        expect = _parse_expectations(raw.get("expect"))
        flags = raw.get("flags") if isinstance(raw.get("flags"), dict) else {}

        cases.append(
            QueryResolutionEvalCase(
                id=case_id,
                prompt=prompt,
                category=category,
                notes=notes,
                history=history,
                expect=expect,
                flags=flags,
            )
        )

    return data, cases


def build_discourse_resolution(
    prompt: str,
    history: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    discourse_enabled: bool = True,
    apply_inference_rewrite: bool = True,
) -> tuple[Any, Any, ResolvedRetrievalQuery]:
    """
    Mirror ``LLMWorker`` discourse + inference rewrite + canonical query build.

    Returns ``(follow_up, discourse_state, resolved_retrieval_query)``.
    """
    from core.discourse_intent import FollowUpClassification, FollowUpKind, classify_follow_up
    from core.discourse_query_rewrite import resolve_ambiguous_user_query
    from core.discourse_state import DiscourseState, update_discourse_state

    hist_list = [dict(h) for h in history]
    raw = (prompt or "").strip()

    if not discourse_enabled:
        follow_up = FollowUpClassification(FollowUpKind.NONE, 0.0)
        resolved = build_resolved_retrieval_query(
            raw_text=raw,
            inference_text=raw,
            follow_up=follow_up,
            discourse=None,
            history=hist_list,
        )
        return follow_up, None, resolved

    prior: DiscourseState | None = None
    if hist_list:
        discourse_state = update_discourse_state(hist_list, prior, prompt)
        follow_up = classify_follow_up(prompt, hist_list, discourse_state)
    else:
        discourse_state = None
        follow_up = FollowUpClassification(FollowUpKind.NONE, 0.0)

    resolved_query = resolve_ambiguous_user_query(
        prompt, discourse_state, follow_up
    )
    inference_text = raw
    if apply_inference_rewrite and resolved_query.succeeded:
        inference_text = resolved_query.resolved

    resolved = build_resolved_retrieval_query(
        raw_text=raw,
        inference_text=inference_text,
        follow_up=follow_up,
        discourse=discourse_state,
        history=hist_list,
        resolved_query=resolved_query,
    )
    return follow_up, discourse_state, resolved


def _check_substrings(
    text: str,
    *,
    must_contain: tuple[str, ...],
    must_not_contain: tuple[str, ...],
    label: str,
) -> list[str]:
    failures: list[str] = []
    blob = text or ""
    blob_lower = blob.lower()
    for needle in must_contain:
        if needle.lower() not in blob_lower:
            failures.append(f"{label}_missing:{needle!r}")
    for needle in must_not_contain:
        if needle.lower() in blob_lower:
            failures.append(f"{label}_forbidden:{needle!r}")
    return failures


def evaluate_resolution_expectations(
    resolved: ResolvedRetrievalQuery,
    expect: QueryResolutionExpectations,
) -> list[str]:
    """Return list of failed check labels (empty when all expectations pass)."""
    failures: list[str] = []
    failures.extend(
        _check_substrings(
            resolved.inference_text,
            must_contain=expect.inference_contains,
            must_not_contain=expect.inference_not_contains,
            label="inference",
        )
    )
    failures.extend(
        _check_substrings(
            resolved.web_text,
            must_contain=expect.web_contains,
            must_not_contain=expect.web_not_contains,
            label="web",
        )
    )
    failures.extend(
        _check_substrings(
            resolved.routing_text,
            must_contain=expect.routing_contains,
            must_not_contain=(),
            label="routing",
        )
    )
    failures.extend(
        _check_substrings(
            resolved.retrieval_text,
            must_contain=expect.retrieval_contains,
            must_not_contain=(),
            label="retrieval",
        )
    )
    return failures


def run_web_fixture_retrieval(
    web_query: str,
    fixture_id: str,
    *,
    fixtures_dir: Path | None = None,
    embed_fn: Callable[[str], Any] | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Replay offline DuckDuckGo HTML and apply the production relevance gate.

    Returns dict with ``web_hits``, ``web_raw_count``, ``web_relevance_dropped``.
    """
    fid = (fixture_id or "").strip()
    if not fid:
        return {"web_hits": 0, "web_raw_count": 0, "web_relevance_dropped": 0}

    base = fixtures_dir or DEFAULT_WEB_FIXTURES_DIR
    path = base / f"{fid}.html"
    if not path.is_file():
        raise FileNotFoundError(f"web fixture not found: {path}")

    raw_items = parse_ddg_html_results(path.read_text(encoding="utf-8"), max_results=max_results)
    query = (web_query or "").strip()
    if not raw_items:
        return {
            "web_hits": 0,
            "web_raw_count": 0,
            "web_relevance_dropped": 0,
        }

    if embed_fn is not None:
        try:
            query_vector = embed_fn(query)
        except Exception as exc:
            logger.debug("web fixture embed failed: %s", exc)
            query_vector = None
        kept, diag = filter_web_results(
            query,
            raw_items,
            query_vector=query_vector,
            embed_text_fn=embed_fn,
            use_embedding_gate=True,
        )
    else:
        kept, diag = filter_web_results(
            query,
            raw_items,
            use_embedding_gate=False,
        )

    dropped = diag.get("web_relevance_dropped") or []
    return {
        "web_hits": len(kept),
        "web_raw_count": len(raw_items),
        "web_relevance_dropped": len(dropped),
        "diagnostics": diag,
    }


def evaluate_query_resolution_case(
    case: QueryResolutionEvalCase,
    *,
    embed_fn: Callable[[str], Any] | None = None,
    fixtures_dir: Path | None = None,
) -> QueryResolutionEvalResult:
    flags = case.flags or {}
    discourse_enabled = bool(flags.get("discourse_enabled", True))
    apply_inference_rewrite = bool(flags.get("apply_inference_rewrite", True))

    try:
        _follow_up, _discourse, resolved = build_discourse_resolution(
            case.prompt,
            case.history,
            discourse_enabled=discourse_enabled,
            apply_inference_rewrite=apply_inference_rewrite,
        )
        failed = evaluate_resolution_expectations(resolved, case.expect)

        fixture_id = case.expect.web_fixture_id
        web_raw = 0
        web_hits = 0
        dropped = 0
        if fixture_id:
            fixture_out = run_web_fixture_retrieval(
                resolved.web_text,
                fixture_id,
                fixtures_dir=fixtures_dir,
                embed_fn=embed_fn,
            )
            web_raw = int(fixture_out.get("web_raw_count") or 0)
            web_hits = int(fixture_out.get("web_hits") or 0)
            dropped = int(fixture_out.get("web_relevance_dropped") or 0)
            if web_hits < case.expect.min_web_hits:
                failed.append(
                    f"web_hits_below_min:{web_hits}<{case.expect.min_web_hits}"
                )

        return QueryResolutionEvalResult(
            case_id=case.id,
            prompt=case.prompt,
            category=case.category,
            notes=case.notes,
            raw_text=resolved.raw_text,
            inference_text=resolved.inference_text,
            routing_text=resolved.routing_text,
            retrieval_text=resolved.retrieval_text,
            web_text=resolved.web_text,
            web_rewrite_reason=resolved.web_rewrite_reason,
            inference_rewrite_applied=resolved.inference_rewritten,
            resolution_pass=not failed,
            failed_checks=failed,
            web_fixture_id=fixture_id,
            web_fixture_raw_count=web_raw,
            web_fixture_hits=web_hits,
            web_relevance_gate_dropped=dropped,
        )
    except Exception as exc:
        logger.exception("query resolution eval failed for %s", case.id)
        return QueryResolutionEvalResult(
            case_id=case.id,
            prompt=case.prompt,
            category=case.category,
            notes=case.notes,
            raw_text=case.prompt,
            inference_text="",
            routing_text="",
            retrieval_text="",
            web_text="",
            web_rewrite_reason="none",
            inference_rewrite_applied=False,
            resolution_pass=False,
            failed_checks=["error"],
            error=str(exc),
        )


def build_query_resolution_summary(
    results: Iterable[QueryResolutionEvalResult],
) -> QueryResolutionEvalSummary:
    rows = list(results)
    total = len(rows)
    passed = sum(1 for r in rows if r.resolution_pass and not r.error)
    failed = total - passed

    fixture_rows = [r for r in rows if r.web_fixture_id]
    fixture_with_hits = [r for r in fixture_rows if r.web_fixture_hits > 0]

    by_category: dict[str, dict[str, Any]] = {}
    for r in rows:
        bucket = by_category.setdefault(
            r.category,
            {"total": 0, "passed": 0, "failed": 0},
        )
        bucket["total"] += 1
        if r.resolution_pass and not r.error:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    failure_rows = [
        {
            "case_id": r.case_id,
            "prompt": r.prompt,
            "failed_checks": r.failed_checks,
            "web_text": r.web_text,
            "inference_text": r.inference_text,
            "error": r.error,
        }
        for r in rows
        if not r.resolution_pass or r.error
    ]

    return QueryResolutionEvalSummary(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=(passed / total if total else 0.0),
        web_fixture_cases=len(fixture_rows),
        web_fixture_hit_rate=(
            len(fixture_with_hits) / len(fixture_rows) if fixture_rows else 0.0
        ),
        by_category=by_category,
        failures=failure_rows,
    )


def format_query_resolution_report(summary: QueryResolutionEvalSummary) -> str:
    lines = [
        "Query resolution evaluation",
        f"  total:     {summary.total}",
        f"  passed:    {summary.passed}",
        f"  failed:    {summary.failed}",
        f"  pass_rate: {summary.pass_rate:.1%}",
        f"  web_fixture_cases: {summary.web_fixture_cases}",
        f"  web_fixture_hit_rate: {summary.web_fixture_hit_rate:.1%}",
    ]
    if summary.failures:
        lines.append("")
        lines.append("Failures:")
        for row in summary.failures[:20]:
            checks = ", ".join(row.get("failed_checks") or [])
            lines.append(f"  - {row['case_id']}: {checks}")
            if row.get("web_text"):
                lines.append(f"      web={row['web_text'][:100]!r}")
    return "\n".join(lines)


def write_query_resolution_run_json(
    path: Path,
    *,
    meta: dict[str, Any],
    summary: QueryResolutionEvalSummary,
    results: list[QueryResolutionEvalResult],
) -> None:
    payload = {
        "schema": "qube.query_resolution_eval_run.v1",
        "meta": meta,
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "QUERY_RESOLUTION_CORPUS_SCHEMA",
    "DEFAULT_WEB_FIXTURES_DIR",
    "QueryResolutionExpectations",
    "QueryResolutionEvalCase",
    "QueryResolutionEvalResult",
    "QueryResolutionEvalSummary",
    "build_discourse_resolution",
    "build_query_resolution_summary",
    "evaluate_query_resolution_case",
    "evaluate_resolution_expectations",
    "expectations_from_flags",
    "format_query_resolution_report",
    "load_query_resolution_corpus",
    "run_web_fixture_retrieval",
    "write_query_resolution_run_json",
]
