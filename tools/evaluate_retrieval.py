#!/usr/bin/env python3
"""Evaluate external knowledge retrieval against a JSON corpus (Phase 2 + Phase 6 Slice 1)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.types import (  # noqa: E402
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
    SERVICE_TRUSTED_KNOWLEDGE,
)
from core.knowledge.scientific_discipline import detect_scientific_discipline  # noqa: E402
from core.knowledge.scientific_discipline_packs import normalize_discipline_id  # noqa: E402
from core.knowledge.http_metrics import format_http_report, merge_http_summaries  # noqa: E402
from core.knowledge.http_throttle_report import (  # noqa: E402
    aggregate_throttle_reports,
    attach_throttle_fields,
)
from core.knowledge.web_retrieval import run_v2_web_retrieval  # noqa: E402

_COVERAGE_RANK = {
    "none": 0,
    "poor": 1,
    "adequate": 2,
    "excellent": 3,
}

_FETCH_RANK = {
    "snippet_only": 0,
    "snippet": 0,
    "abstract": 1,
    "full_text": 2,
}

_SERVICE_BY_NAME = {
    "scientific_evidence": SERVICE_SCIENTIFIC_EVIDENCE,
    "trusted_knowledge": SERVICE_TRUSTED_KNOWLEDGE,
    "finance_knowledge": SERVICE_FINANCE_KNOWLEDGE,
    "legal_knowledge": SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE: SERVICE_SCIENTIFIC_EVIDENCE,
    SERVICE_TRUSTED_KNOWLEDGE: SERVICE_TRUSTED_KNOWLEDGE,
    SERVICE_FINANCE_KNOWLEDGE: SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE: SERVICE_LEGAL_KNOWLEDGE,
}

_DEFAULT_CORPUS = {
    SERVICE_SCIENTIFIC_EVIDENCE: ROOT / "eval" / "retrieval_corpus" / "v1_scientific.json",
    SERVICE_TRUSTED_KNOWLEDGE: ROOT / "eval" / "retrieval_corpus" / "v1_trusted.json",
    SERVICE_FINANCE_KNOWLEDGE: ROOT / "eval" / "retrieval_corpus" / "v1_finance.json",
    SERVICE_LEGAL_KNOWLEDGE: ROOT / "eval" / "retrieval_corpus" / "v1_legal.json",
}


def _discipline_tag(entry: dict) -> str:
    return str(entry.get("discipline") or entry.get("expect_discipline") or "").strip()


def _discipline_primary_stats(
    entries: list[dict],
    results: list[dict],
) -> dict[str, dict[str, int | float]]:
    """Primary-adapter hit rate grouped by corpus ``discipline`` tag."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for entry, result in zip(entries, results):
        discipline = _discipline_tag(entry)
        primary = str(entry.get("primary_adapter") or "").strip().lower()
        if not discipline or not primary:
            continue
        primary_ok = bool(result.get("checks", {}).get("primary_ok"))
        groups[discipline].append(primary_ok)

    stats: dict[str, dict[str, int | float]] = {}
    for discipline, hits in sorted(groups.items()):
        total = len(hits)
        primary_hits = sum(1 for ok in hits if ok)
        stats[discipline] = {
            "total": total,
            "primary_hits": primary_hits,
            "primary_rate": round(primary_hits / total, 3) if total else 0.0,
        }
    return stats


def _groups_below_threshold(
    stats: dict[str, dict[str, int | float]],
    *,
    threshold: float,
) -> list[str]:
    failing: list[str] = []
    for discipline, row in stats.items():
        rate = float(row.get("primary_rate") or 0.0)
        if rate < threshold:
            failing.append(discipline)
    return failing


@contextmanager
def _scientific_eval_preferences(*, use_user_prefs: bool):
    """Use catalog defaults during scientific eval unless ``use_user_prefs``."""
    if use_user_prefs:
        yield
        return
    targets = (
        "core.app_settings.get_knowledge_source_preferences",
        "core.knowledge.pipeline_scientific.get_knowledge_source_preferences",
    )
    with patch(targets[0], return_value={}), patch(targets[1], return_value={}):
        yield


def _inter_query_delay_s(knowledge_service: str) -> float:
    if knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
        return 2.0
    if knowledge_service in {SERVICE_TRUSTED_KNOWLEDGE, SERVICE_FINANCE_KNOWLEDGE}:
        return 1.2
    return 0.0


def _evaluate_with_optional_retry(
    entry: dict,
    *,
    live: bool,
    knowledge_service: str,
    adapter_filter: tuple[str, ...] | None = None,
) -> dict:
    result = _evaluate_query(
        entry,
        live=live,
        knowledge_service=knowledge_service,
        adapter_filter=adapter_filter,
    )
    if (
        live
        and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE
        and result.get("status") == "no_results"
    ):
        time.sleep(3.0)
        result = _evaluate_query(
            entry,
            live=live,
            knowledge_service=knowledge_service,
            adapter_filter=adapter_filter,
        )
    return result


def _load_corpus(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, list(data.get("queries") or [])


def _resolve_service(corpus_meta: dict, override: str | None) -> str:
    if override:
        key = override.strip().lower()
        if key not in _SERVICE_BY_NAME:
            raise ValueError(f"Unknown service: {override}")
        return _SERVICE_BY_NAME[key]
    service = str(corpus_meta.get("service") or SERVICE_SCIENTIFIC_EVIDENCE)
    return _SERVICE_BY_NAME.get(service, service)


def _best_fetch_rank(sources) -> str:
    ranks = [_FETCH_RANK.get(str(s.fetch_status or ""), 0) for s in sources]
    if not ranks:
        return "none"
    best = max(ranks)
    for label, value in _FETCH_RANK.items():
        if value == best:
            return label
    return "snippet_only"


def _resolve_adapter_filter(
    entry: dict,
    *,
    cli_adapters: tuple[str, ...] | None,
    single_adapter: bool,
) -> tuple[str, ...] | None:
    """Corpus entry override, CLI flag, or --single-adapter from expect_adapters."""
    forced = entry.get("force_adapter") or entry.get("force_adapters")
    if forced:
        if isinstance(forced, str):
            ids = (forced.strip().lower(),)
        else:
            ids = tuple(str(a).strip().lower() for a in forced if str(a).strip())
        return ids or None
    if cli_adapters:
        return cli_adapters
    if single_adapter:
        expect = entry.get("expect_adapters") or []
        if len(expect) == 1:
            return (str(expect[0]).strip().lower(),)
    return None


def _evaluate_query(
    entry: dict,
    *,
    live: bool,
    knowledge_service: str,
    adapter_filter: tuple[str, ...] | None = None,
) -> dict:
    query = str(entry.get("query") or "").strip()
    if not query:
        return {"id": entry.get("id"), "status": "skipped", "reason": "empty_query"}

    if not live:
        payload = {"id": entry.get("id"), "query": query, "status": "dry_run"}
        if knowledge_service == SERVICE_TRUSTED_KNOWLEDGE:
            payload["trusted_criteria"] = True
        return payload

    outcome = run_v2_web_retrieval(
        query=query,
        semantic_query=query,
        knowledge_service=knowledge_service,
        adapter_filter=adapter_filter,
    )
    bundle = outcome.bundle
    if bundle is None or not bundle.sources:
        payload = {
            "id": entry.get("id"),
            "query": query,
            "status": "no_results",
            "latency_ms": round(outcome.latency_ms, 1),
            "knowledge_service": knowledge_service,
        }
        if outcome.relevance_diag and outcome.relevance_diag.get("http_summary"):
            payload["http_summary"] = outcome.relevance_diag["http_summary"]
        return attach_throttle_fields(payload)

    sources = bundle.sources
    adapters = {s.adapter for s in sources}
    abstract_hits = sum(1 for s in sources if s.fetch_status == "abstract")
    max_authority = max(s.authority_score for s in sources)
    has_wikipedia = any(s.adapter == "wikipedia_api" for s in sources)
    fetch_rank = _best_fetch_rank(sources)

    expect = set(entry.get("expect_adapters") or [])
    adapter_ok = bool(expect.intersection(adapters)) if expect else True

    min_sources = max(1, int(entry.get("min_sources") or 1))
    sources_ok = len(sources) >= min_sources

    expect_abstract = bool(entry.get("expect_abstract", False))
    abstract_ok = (not expect_abstract) or abstract_hits >= 1

    min_authority = entry.get("min_authority")
    authority_ok = (
        min_authority is None or max_authority >= float(min_authority)
    )

    require_wikipedia = bool(entry.get("require_wikipedia", False))
    wikipedia_ok = (not require_wikipedia) or has_wikipedia

    min_fetch = str(entry.get("min_fetch_rank") or "").strip().lower()
    fetch_ok = (
        not min_fetch
        or _FETCH_RANK.get(fetch_rank, 0) >= _FETCH_RANK.get(min_fetch, 0)
    )

    min_cov = str(entry.get("min_coverage_rank") or "").lower()
    cov_ok = (
        not min_cov
        or _COVERAGE_RANK.get(bundle.coverage, 0)
        >= _COVERAGE_RANK.get(min_cov, 0)
    )

    require_warning = str(entry.get("require_warning") or "").strip()
    warning_ok = (not require_warning) or require_warning in (bundle.warnings or ())

    expect_discipline = str(entry.get("discipline") or entry.get("expect_discipline") or "").strip()
    detected_discipline = None
    discipline_ok = True
    if expect_discipline and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
        detected_discipline = detect_scientific_discipline(query).discipline
        discipline_ok = normalize_discipline_id(detected_discipline) == normalize_discipline_id(
            expect_discipline
        )

    primary_adapter = str(entry.get("primary_adapter") or "").strip().lower()
    primary_ok = True
    if primary_adapter:
        primary_ok = primary_adapter in {a.lower() for a in adapters}

    checks_ok = all(
        (
            adapter_ok,
            sources_ok,
            abstract_ok,
            authority_ok,
            wikipedia_ok,
            fetch_ok,
            cov_ok,
            warning_ok,
            discipline_ok,
            primary_ok,
        )
    )
    status = "ok" if checks_ok else "partial"

    payload = {
        "id": entry.get("id"),
        "query": query,
        "status": status,
        "latency_ms": round(outcome.latency_ms, 1),
        "knowledge_service": knowledge_service,
        "adapters": sorted(adapters),
        "source_count": len(sources),
        "abstract_hits": abstract_hits,
        "max_authority": round(max_authority, 3),
        "has_wikipedia": has_wikipedia,
        "best_fetch_rank": fetch_rank,
        "coverage": bundle.coverage,
        "confidence": round(bundle.confidence, 3),
        "stop_reason": bundle.stop_reason,
        "titles": [s.title for s in sources[:5]],
        "checks": {
            "adapter_ok": adapter_ok,
            "sources_ok": sources_ok,
            "abstract_ok": abstract_ok,
            "authority_ok": authority_ok,
            "wikipedia_ok": wikipedia_ok,
            "fetch_ok": fetch_ok,
            "coverage_ok": cov_ok,
            "warning_ok": warning_ok,
            "discipline_ok": discipline_ok,
            "primary_ok": primary_ok,
        },
    }
    if detected_discipline is not None:
        payload["detected_discipline"] = detected_discipline
        payload["expect_discipline"] = expect_discipline
    if primary_adapter:
        payload["primary_adapter"] = primary_adapter
    if adapter_filter:
        payload["adapter_filter"] = list(adapter_filter)
    if outcome.relevance_diag and outcome.relevance_diag.get("http_summary"):
        payload["http_summary"] = outcome.relevance_diag["http_summary"]
    return attach_throttle_fields(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate knowledge retrieval corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus JSON path (default: v1_scientific or v1_trusted by --service)",
    )
    parser.add_argument(
        "--service",
        choices=["scientific_evidence", "trusted_knowledge", "finance_knowledge", "legal_knowledge"],
        default=None,
        help="Knowledge service (overrides corpus service field when set with --corpus)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live adapter calls (requires network)",
    )
    parser.add_argument(
        "--min-pass",
        type=int,
        default=None,
        help="Exit non-zero unless at least N entries status=ok (live only)",
    )
    parser.add_argument(
        "--min-discipline-primary-rate",
        type=float,
        default=None,
        help="Exit non-zero unless discipline-tagged queries hit primary_adapter at this rate (live only)",
    )
    parser.add_argument(
        "--min-discipline-group-primary-rate",
        type=float,
        default=None,
        help="Exit non-zero unless every discipline group meets this primary rate (live only)",
    )
    parser.add_argument(
        "--user-prefs",
        action="store_true",
        help="Use saved knowledge source preferences instead of catalog defaults (scientific eval)",
    )
    parser.add_argument(
        "--http-report",
        action="store_true",
        help="Include aggregated HTTP metrics in JSON stdout (live mode)",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        dest="adapters",
        metavar="ADAPTER_ID",
        help="Force retrieval through only this adapter (repeatable; bypasses routing)",
    )
    parser.add_argument(
        "--single-adapter",
        action="store_true",
        help="When a corpus row has exactly one expect_adapters entry, force that adapter",
    )
    args = parser.parse_args()

    cli_adapter_filter: tuple[str, ...] | None = None
    if args.adapters:
        cli_adapter_filter = tuple(
            str(a).strip().lower() for a in args.adapters if str(a).strip()
        )

    corpus_path = args.corpus
    if corpus_path is None:
        service_key = args.service or SERVICE_SCIENTIFIC_EVIDENCE
        corpus_path = _DEFAULT_CORPUS.get(
            _SERVICE_BY_NAME.get(service_key, service_key),
            _DEFAULT_CORPUS[SERVICE_SCIENTIFIC_EVIDENCE],
        )

    corpus_meta, entries = _load_corpus(corpus_path)
    knowledge_service = _resolve_service(corpus_meta, args.service)

    results = []
    pref_ctx = (
        _scientific_eval_preferences(use_user_prefs=args.user_prefs)
        if args.live and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE
        else nullcontext()
    )
    with pref_ctx:
        for i, entry in enumerate(entries):
            if args.live and i > 0:
                delay = _inter_query_delay_s(knowledge_service)
                if delay > 0:
                    time.sleep(delay)
            results.append(
                _evaluate_with_optional_retry(
                    entry,
                    live=args.live,
                    knowledge_service=knowledge_service,
                    adapter_filter=_resolve_adapter_filter(
                        entry,
                        cli_adapters=cli_adapter_filter,
                        single_adapter=args.single_adapter,
                    ),
                )
            )
    ok = sum(1 for r in results if r.get("status") == "ok")
    partial = sum(1 for r in results if r.get("status") == "partial")
    dry = sum(1 for r in results if r.get("status") == "dry_run")
    discipline_tagged = [
        (entry, result)
        for entry, result in zip(entries, results)
        if str(entry.get("discipline") or entry.get("expect_discipline") or "").strip()
        and str(entry.get("primary_adapter") or "").strip()
    ]
    primary_hits = sum(
        1
        for _entry, result in discipline_tagged
        if result.get("checks", {}).get("primary_ok")
    )
    primary_rate = (
        primary_hits / len(discipline_tagged) if discipline_tagged else None
    )
    discipline_stats = _discipline_primary_stats(entries, results)
    summary = {
        "corpus": str(corpus_path),
        "service": knowledge_service,
        "results": results,
        "ok": ok,
        "partial": partial,
        "dry_run": dry,
        "total": len(results),
    }
    if cli_adapter_filter:
        summary["forced_adapters"] = list(cli_adapter_filter)
    if args.single_adapter:
        summary["single_adapter_mode"] = True
    if primary_rate is not None:
        summary["discipline_primary_hits"] = primary_hits
        summary["discipline_primary_total"] = len(discipline_tagged)
        summary["discipline_primary_rate"] = round(primary_rate, 3)
    if discipline_stats:
        summary["discipline_group_primary"] = discipline_stats
    if args.live:
        http_summaries = [r.get("http_summary") for r in results if r.get("http_summary")]
        if http_summaries:
            summary["http_report"] = merge_http_summaries(http_summaries)
        throttle_reports = [r.get("throttle_report") for r in results if r.get("throttle_report")]
        if throttle_reports:
            summary["throttle_report"] = aggregate_throttle_reports(throttle_reports)
        failure_classes = [
            r.get("failure_class")
            for r in results
            if r.get("failure_class")
        ]
        if failure_classes:
            summary["failure_class_counts"] = {
                "retrieval": sum(1 for c in failure_classes if c == "retrieval"),
                "throttle": sum(1 for c in failure_classes if c == "throttle"),
                "mixed": sum(1 for c in failure_classes if c == "mixed"),
            }
    elif args.http_report:
        http_summaries = [r.get("http_summary") for r in results if r.get("http_summary")]
        if http_summaries:
            summary["http_report"] = merge_http_summaries(http_summaries)
    print(json.dumps(summary, indent=2))

    if not args.live:
        return 0

    min_pass = args.min_pass
    if min_pass is None:
        if knowledge_service == SERVICE_TRUSTED_KNOWLEDGE:
            min_pass = max(1, len(results) - 1)  # ≥ 4/5 default
        elif knowledge_service == SERVICE_FINANCE_KNOWLEDGE:
            min_pass = max(3, len(results) - 1)  # ≥ 3/4 default
        else:
            min_pass = len(results)  # 5/5 default for scientific

    print(
        f"\nSummary: {ok}/{len(results)} ok, {partial} partial (min_pass={min_pass})",
        file=sys.stderr,
    )
    if primary_rate is not None:
        min_primary = args.min_discipline_primary_rate
        if min_primary is None and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
            min_primary = 0.7
        if min_primary is not None:
            print(
                f"Discipline primary adapter: {primary_hits}/{len(discipline_tagged)} "
                f"({primary_rate:.1%}, min={min_primary:.0%})",
                file=sys.stderr,
            )
    if discipline_stats:
        min_group = args.min_discipline_group_primary_rate
        if min_group is None and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
            min_group = 0.7
        if min_group is not None:
            for discipline, row in sorted(discipline_stats.items()):
                rate = float(row["primary_rate"])
                print(
                    f"  {discipline}: {row['primary_hits']}/{row['total']} "
                    f"({rate:.1%})",
                    file=sys.stderr,
                )
            failing = _groups_below_threshold(discipline_stats, threshold=min_group)
            if failing:
                print(
                    f"Discipline groups below {min_group:.0%}: {', '.join(failing)}",
                    file=sys.stderr,
                )
    http_report = summary.get("http_report")
    if http_report:
        print(f"\n{format_http_report(http_report)}", file=sys.stderr)
    throttle_report = summary.get("throttle_report")
    if throttle_report:
        print(
            "\nThrottle: "
            f"{throttle_report.get('queries_throttled', 0)} queries with HTTP pressure, "
            f"{throttle_report.get('queries_short_circuited', 0)} short-circuited",
            file=sys.stderr,
        )
        failure_counts = summary.get("failure_class_counts") or {}
        if failure_counts:
            print(
                "Failure classes: "
                f"retrieval={failure_counts.get('retrieval', 0)}, "
                f"throttle={failure_counts.get('throttle', 0)}, "
                f"mixed={failure_counts.get('mixed', 0)}",
                file=sys.stderr,
            )
        hosts_open = throttle_report.get("hosts_open") or []
        if hosts_open:
            print(f"  Open circuits: {', '.join(hosts_open)}", file=sys.stderr)
    if ok < min_pass:
        return 1
    min_primary = args.min_discipline_primary_rate
    if min_primary is None and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
        min_primary = 0.7
    if (
        min_primary is not None
        and primary_rate is not None
        and primary_rate < min_primary
    ):
        return 1
    min_group = args.min_discipline_group_primary_rate
    if min_group is None and knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
        min_group = 0.7
    if min_group is not None and discipline_stats:
        if _groups_below_threshold(discipline_stats, threshold=min_group):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
