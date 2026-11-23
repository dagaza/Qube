#!/usr/bin/env python3
"""Capture live retrieval candidates into knowledge-quality replay fixtures (Stage S2).

For each corpus case with a ``path`` the tool runs the same adapter / discovery
entry points the production pipeline for that path uses, for up to two query
variants:

* ``production`` — what today's code sends (composer token stripped, then
  ``core.query_resolution_evaluation.build_discourse_resolution`` for multi-turn
  cases, which mirrors ``LLMWorker`` discourse resolution);
* ``standalone`` — the case's gold standalone question (what S5/S6 aim for).

Rows are captured deeper than production uses (``--depth``) and each records
where it came from (``retrieved``: query variant, adapter, rank). The fixture's
``production_depth`` lets the evaluator rebuild the exact production pool, while
the full capture supports recall-widening work (S8).

Fixtures are written **without any scores** and with ``label: null`` so labelling
stays blind to rankers. Existing fixtures are never overwritten without
``--force``; labels from an existing file are carried over for rows that are
still present when ``--keep-labels`` is given.

Usage::

    .venv/bin/python tools/capture_knowledge_fixture.py --case kq_ace_hf_single
    .venv/bin/python tools/capture_knowledge_fixture.py --all-missing
    .venv/bin/python tools/capture_knowledge_fixture.py --from-evidence-cache \\
        ~/.qube/evidence_cache/<key>.json --case kq_ant_resolved
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS = ROOT / "eval" / "retrieval_corpus" / "v1_knowledge_quality.json"
FULL_TEXT_CAP = 4000
# Pause between calls to the same provider (arXiv asks for ~3 s; DDG throttles).
# Qube's own DDG burst budget is 6 live queries / 10 min (core/knowledge/discovery/
# session_budget.py); web captures must pace under it (>= 100 s) or fall back.
_PROVIDER_PAUSE_S = {"arxiv": 3.2, "duckduckgo": 4.0, "discovery": 110.0}
_last_call: dict[str, float] = {}
# Discovery provider used by the most recent web capture (None for adapter-only paths).
_last_discovery: dict[str, str | None] = {"provider": None, "primary": None}


# Outcomes that mean the provider did not really answer (vs. a genuine empty SERP).
_DEGRADED_OUTCOMES = {"bot_challenge", "empty_parse", "network_error"}


def _record_discovery(discovery) -> None:
    """Remember which provider answered and whether the answer was a real SERP.

    A genuine "no results" page (expected for fictional-entity abstain cases)
    is *not* degraded; a fallback provider, a bot challenge, a parse failure or
    Qube's own DuckDuckGo burst limiter tripping is.
    """
    from core.knowledge.discovery.privacy_policy import resolve_discovery_route

    provider = discovery.provider_id or "duckduckgo"
    outcome = discovery.search_outcome
    kind = str(getattr(getattr(outcome, "kind", None), "value", "") or "")
    if kind in _DEGRADED_OUTCOMES:
        provider = f"{provider}:{kind}"
    _last_discovery["provider"] = provider
    _last_discovery["primary"] = resolve_discovery_route().primary_id


def _pace(provider: str) -> None:
    wait = _PROVIDER_PAUSE_S.get(provider, 1.0)
    last = _last_call.get(provider)
    if last is not None:
        remaining = wait - (time.monotonic() - last)
        if remaining > 0:
            time.sleep(remaining)
    _last_call[provider] = time.monotonic()


def _clean_row(row: dict[str, Any], adapter: str) -> dict[str, Any]:
    """Keep content/metadata; drop every private/score key (``_…``) but ``_adapter``."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_") and key != "_adapter":
            continue
        out[key] = value
    out["_adapter"] = str(row.get("_adapter") or adapter)
    full = out.get("full_text")
    if isinstance(full, str) and len(full) > FULL_TEXT_CAP:
        out["full_text"] = full[:FULL_TEXT_CAP]
    return out


def _row_key(row: dict[str, Any]) -> str:
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    url = str(row.get("url") or "").strip().lower().rstrip("/")
    if url:
        return f"url:{url}"
    return "title:" + " ".join(str(row.get("title") or "").lower().split())


def production_query_for(entry: dict[str, Any]) -> str:
    from core.prompt_history import strip_composer_tokens_from_history
    from core.query_resolution_evaluation import build_discourse_resolution

    raw = str(entry.get("query") or "").strip()
    cleaned = strip_composer_tokens_from_history([{"role": "user", "content": raw}])
    query = str(cleaned[0].get("content") or raw).strip()
    history = list(entry.get("history") or [])
    _, _, resolved = build_discourse_resolution(query, history)
    return (resolved.web_text or query).strip()


# --- per-path capture -------------------------------------------------------


def _capture_scientific(query: str, depth: int) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, str]]:
    from core.app_settings import get_knowledge_source_preferences
    from core.knowledge.adapters.registry import get_search_function
    from core.knowledge.retrieval_profiles import (
        get_profile_spec,
        normalize_profile_id,
        order_adapter_ids,
    )
    from core.knowledge.scientific_query_planner import adapter_query_for, plan_scientific_query
    from core.knowledge.source_preferences import resolve_service_adapters
    from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

    plan = plan_scientific_query(query, semantic_query=query)
    profile = get_profile_spec(normalize_profile_id("balanced"))
    adapter_ids = order_adapter_ids(
        tuple(
            resolve_service_adapters(
                SERVICE_SCIENTIFIC_EVIDENCE,
                query=query,
                composer_adapter_filter=None,
                stored_preferences=get_knowledge_source_preferences(),
            )
        ),
        profile=profile,
    )
    budget = profile.budget.max_results
    per_adapter_prod = max(2, budget)  # pipeline_scientific: per_adapter = max(2, budget)
    results: dict[str, list[dict]] = {}
    adapter_queries: dict[str, str] = {}
    for aid in adapter_ids:
        fn = get_search_function(aid)
        if fn is None:
            continue
        q = adapter_query_for(plan, aid)
        adapter_queries[aid] = q
        _pace(aid)
        try:
            rows = fn(q, max_results=depth) or []
        except Exception as exc:  # adapter failures are recorded, not fatal
            print(f"    ! {aid} failed: {exc}", file=sys.stderr)
            rows = []
        results[aid] = [_clean_row(r, aid) for r in rows if isinstance(r, dict)]
    return results, {aid: per_adapter_prod for aid in results}, adapter_queries


def _discovery_rows(discovery) -> list[dict[str, Any]]:
    rows = [
        {"title": c.title or "", "snippet": c.snippet or "", "url": c.url}
        for c in discovery.candidates
    ]
    if not rows and discovery.raw_rows:
        rows = [dict(r) for r in discovery.raw_rows if isinstance(r, dict)]
    return rows


def _capture_general_web(query: str, depth: int):
    from core.knowledge.adapters.duckduckgo import is_failure_sentinel
    from core.knowledge.discovery import discover_full
    from core.knowledge.retrieval_profiles import get_profile_spec

    profile = get_profile_spec("balanced")
    # pipeline_general_web._discovery_max_results
    prod = min(max(profile.budget.max_results, profile.fetch_url_count * 2, 3), 8)
    _pace("discovery")
    discovery = discover_full(query, max_results=max(depth, prod), retrieval_profile="balanced")
    _record_discovery(discovery)
    rows = _discovery_rows(discovery)
    if is_failure_sentinel(rows):
        print(f"    ! discovery failure sentinel for {query!r}", file=sys.stderr)
        rows = []
    provider = discovery.provider_id or "duckduckgo"
    return {provider: [_clean_row(r, provider) for r in rows]}, {provider: prod}, {provider: query}


def _capture_trusted(query: str, depth: int):
    from core.knowledge.adapters.duckduckgo import is_failure_sentinel
    from core.knowledge.adapters.wikipedia_api import ADAPTER_ID as WIKI, search_wikipedia
    from core.knowledge.discovery import discover_full_with_fallback
    from core.knowledge.retrieval_profiles import get_profile_spec

    budget = get_profile_spec("balanced").budget.max_results
    _pace(WIKI)
    wiki = search_wikipedia(query, max_results=max(2, min(depth, 4))) or []
    _pace("discovery")
    discovery = discover_full_with_fallback(
        query, max_results=max(depth, 5, budget * 2), retrieval_profile="balanced"
    )
    _record_discovery(discovery)
    serp = _discovery_rows(discovery)
    if is_failure_sentinel(serp):
        print(f"    ! discovery failure sentinel for {query!r}", file=sys.stderr)
        serp = []
    provider = discovery.provider_id or "duckduckgo"
    results = {
        WIKI: [_clean_row(r, WIKI) for r in wiki if isinstance(r, dict)],
        provider: [_clean_row(r, provider) for r in serp],
    }
    depths = {WIKI: min(2, budget), provider: max(5, budget * 2)}  # pipeline_trusted
    return results, depths, {WIKI: query, provider: query}


def _capture_finance(query: str, depth: int):
    from core.app_settings import get_knowledge_source_preferences
    from core.knowledge.finance_query_planner import plan_finance_query
    from core.knowledge.pipeline_finance import _fetch_finance_rows
    from core.knowledge.retrieval_profiles import get_profile_spec
    from core.knowledge.source_preferences import resolve_service_adapters
    from core.knowledge.types import SERVICE_FINANCE_KNOWLEDGE

    budget = get_profile_spec("balanced").budget.max_results
    plan = plan_finance_query(query, semantic_query=query)
    adapter_ids = resolve_service_adapters(
        SERVICE_FINANCE_KNOWLEDGE,
        query=query,
        composer_adapter_filter=None,
        stored_preferences=get_knowledge_source_preferences(),
    )
    results: dict[str, list[dict]] = {}
    for aid in adapter_ids:
        _pace(aid)
        _, rows = _fetch_finance_rows(
            adapter_ids=(aid,),
            search_query=plan.search_query,
            form_filter=plan.form_types,
            max_results=depth,
        )
        results[aid] = [_clean_row(r, aid) for r in rows if isinstance(r, dict)]
    prod = max(2, max(3, budget))  # pipeline_finance: per_adapter = max(2, max(3, budget))
    return results, {aid: prod for aid in results}, {aid: plan.search_query for aid in results}


_CAPTURERS = {
    "scientific_evidence": _capture_scientific,
    "general_web": _capture_general_web,
    "trusted_knowledge": _capture_trusted,
    "finance_knowledge": _capture_finance,
}


def capture_case(entry: dict[str, Any], *, depth: int) -> dict[str, Any]:
    path = str(entry.get("path") or "").strip()
    capturer = _CAPTURERS.get(path)
    if capturer is None:
        raise SystemExit(f"{entry.get('id')}: no capturer for path {path!r}")

    variants = {"production": production_query_for(entry)}
    standalone = str(entry.get("standalone_question") or "").strip()
    if standalone and standalone.lower() != variants["production"].lower():
        variants["standalone"] = standalone

    candidates: dict[str, dict[str, Any]] = {}
    production_depth: dict[str, int] = {}
    adapter_queries: dict[str, dict[str, str]] = {}
    discovery_provider: dict[str, str] = {}
    degraded: list[str] = []
    for variant, query in variants.items():
        print(f"  [{variant}] {query!r}")
        _last_discovery.update(provider=None, primary=None)
        results, depths, aq = capturer(query, depth)
        if _last_discovery["provider"]:
            discovery_provider[variant] = _last_discovery["provider"]
            if _last_discovery["provider"] != _last_discovery["primary"]:
                degraded.append(variant)
                print(
                    f"    ! discovery degraded: {_last_discovery['primary']} -> "
                    f"{_last_discovery['provider']}",
                    file=sys.stderr,
                )
        production_depth.update(depths)
        adapter_queries[variant] = aq
        for adapter, rows in results.items():
            print(f"    {adapter}: {len(rows)} rows")
            for rank, row in enumerate(rows):
                key = _row_key(row)
                cand = candidates.get(key)
                if cand is None:
                    cand = dict(row)
                    cand["retrieved"] = []
                    cand["label"] = None
                    candidates[key] = cand
                cand["retrieved"].append({"query": variant, "adapter": adapter, "rank": rank})

    raw = str(entry.get("query") or "").strip()
    tool = str(entry.get("tool") or "").strip()
    fixture = {
        "schema_version": 2,
        "id": entry["id"],
        "path": path,
        "category": entry.get("category") or "uncategorized",
        "description": entry.get("notes") or "",
        "query_raw": f"@[tool:{tool}] {raw}" if tool and not raw.startswith("@[") else raw,
        "query_clean": raw,
        "retrieval_query": variants["production"],
        "standalone_question": standalone or None,
        "gold_subject": entry.get("gold_subject"),
        "gold_facet": entry.get("gold_facet"),
        "budget": 3,
        "history": list(entry.get("history") or []),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "capture_profile": "balanced",
        "production_depth": production_depth,
        "adapter_queries": adapter_queries,
        "discovery_provider": discovery_provider or None,
        # Variants whose web discovery fell back from the primary provider
        # (e.g. DuckDuckGo bot challenge -> Wikipedia). Such captures do not
        # represent the production pool and must be re-captured (V15).
        "capture_degraded": degraded or None,
        "candidates": list(candidates.values()),
    }
    return {k: v for k, v in fixture.items() if v is not None}


def import_evidence_cache(entry: dict[str, Any], cache_file: Path) -> dict[str, Any]:
    """Build a fixture from rows the app cached (``~/.qube/evidence_cache``)."""
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    rows = data.get("rows") if isinstance(data, dict) else data
    candidates: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        cand = _clean_row(row, str(row.get("_adapter") or "unknown"))
        cand["retrieved"] = [{"query": "production", "adapter": cand["_adapter"], "rank": rank}]
        cand["label"] = None
        candidates.setdefault(_row_key(cand), cand)
    raw = str(entry.get("query") or "").strip()
    return {
        "schema_version": 2,
        "id": entry["id"],
        "path": entry.get("path") or "scientific_evidence",
        "category": entry.get("category") or "uncategorized",
        "description": (entry.get("notes") or "") + f" Imported from evidence cache {cache_file.name}.",
        "query_raw": raw,
        "query_clean": raw,
        "retrieval_query": raw,
        "standalone_question": entry.get("standalone_question"),
        "budget": 3,
        "history": list(entry.get("history") or []),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidates": list(candidates.values()),
    }


def _carry_labels(new: dict[str, Any], old_path: Path) -> int:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    labels = {
        _row_key(c): (c.get("label"), c.get("label_source"), c.get("synthetic"))
        for c in old.get("candidates") or []
        if isinstance(c, dict)
    }
    carried = 0
    for cand in new["candidates"]:
        hit = labels.get(_row_key(cand))
        if hit and hit[0]:
            cand["label"], cand["label_source"] = hit[0], hit[1]
            carried += 1
    return carried


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--case", action="append", default=[], help="Corpus case id (repeatable)")
    parser.add_argument("--all-missing", action="store_true", help="Capture every case whose fixture file is missing")
    parser.add_argument("--depth", type=int, default=8, help="Rows per adapter/provider to capture")
    parser.add_argument("--from-evidence-cache", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing fixture files")
    parser.add_argument("--keep-labels", action="store_true", help="Carry labels over from the existing file")
    parser.add_argument(
        "--discovery-pause", type=float, default=_PROVIDER_PAUSE_S["discovery"],
        help="Seconds between web discovery requests (be gentle with DuckDuckGo)",
    )
    parser.add_argument("--path", action="append", default=[], help="Only cases on this path (repeatable)")
    parser.add_argument(
        "--redo-degraded", action="store_true",
        help="Select web/trusted cases whose fixture is missing, degraded, or predates provider tracking",
    )
    args = parser.parse_args()
    _PROVIDER_PAUSE_S["discovery"] = max(0.0, args.discovery_pause)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in corpus.get("queries") or [] if e.get("id")}
    selected = [entries[c] for c in args.case if c in entries]
    missing_ids = [c for c in args.case if c not in entries]
    if missing_ids:
        print(f"unknown case ids: {missing_ids}", file=sys.stderr)
        return 2
    if args.all_missing:
        selected += [
            e for e in entries.values()
            if e.get("fixture") and e.get("path") and not (ROOT / e["fixture"]).exists()
            and e not in selected
        ]
    if args.redo_degraded:
        args.force = True
        for e in entries.values():
            if e.get("path") not in ("general_web", "trusted_knowledge") or e in selected:
                continue
            fx_path = ROOT / str(e.get("fixture") or "")
            if not fx_path.is_file():
                selected.append(e)
                continue
            fx = json.loads(fx_path.read_text(encoding="utf-8"))
            if fx.get("capture_degraded") or not fx.get("discovery_provider"):
                selected.append(e)
    if args.path:
        if not selected:  # --path alone selects every case on those paths
            selected = list(entries.values())
        selected = [e for e in selected if e.get("path") in set(args.path)]
    if not selected:
        print("nothing to capture", file=sys.stderr)
        return 2

    for entry in selected:
        out = ROOT / str(entry.get("fixture") or f"eval/fixtures/knowledge_quality/{entry['id']}.json")
        if out.exists() and not (args.force or args.keep_labels):
            print(f"skip {entry['id']}: {out.name} exists (use --force)")
            continue
        print(f"capture {entry['id']} ({entry.get('path')})")
        if args.from_evidence_cache:
            fixture = import_evidence_cache(entry, args.from_evidence_cache.expanduser())
        else:
            fixture = capture_case(entry, depth=args.depth)
        if out.exists() and args.keep_labels:
            print(f"  carried {_carry_labels(fixture, out)} labels")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  wrote {out.relative_to(ROOT)} ({len(fixture['candidates'])} candidates)")
        if fixture.get("capture_degraded"):
            # Stop instead of hammering a provider that is challenging us; the
            # degraded fixture fails validation until it is re-captured.
            print("  discovery degraded — stopping; re-run later with --force", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
