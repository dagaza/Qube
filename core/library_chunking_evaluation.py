"""
Offline Library RAG / chunking evaluation harness (Phase 0 lite).

Pure evaluation logic (no Qt). Used by ``tools/evaluate_library_chunking.py`` and unit tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger("Qube.LibraryEval")

CORPUS_SCHEMA = "qube.library_corpus.v1"
RUN_SCHEMA = "qube.library_eval_run.v1"
EVAL_LIBRARY_PREFIX = "eval_"


@dataclass
class LibraryEvalConfig:
    top_k: int = 5
    duplicate_jaccard_threshold: float = 0.75


@dataclass
class LibraryEvalResult:
    case_id: str
    query: str
    category: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit_count: int
    recall_at_k: float
    reciprocal_rank: float
    substring_success: bool
    duplicate_pair_rate: float
    unique_chunk_rate: float
    success: bool
    failure_reason: str
    expect_contains: list[str] = field(default_factory=list)
    missing_substrings: list[str] = field(default_factory=list)
    forbidden_source_hits: list[str] = field(default_factory=list)
    notes: str = ""


def estimate_tokens(text: str) -> float:
    return len((text or "").strip()) / 4.0


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def duplicate_pair_rate(
    texts: list[str],
    *,
    threshold: float = 0.75,
) -> float:
    """Share of unordered text pairs in the result set above the Jaccard threshold."""
    if len(texts) < 2:
        return 0.0
    dup_pairs = 0
    total_pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total_pairs += 1
            if jaccard_similarity(texts[i], texts[j]) >= threshold:
                dup_pairs += 1
    return dup_pairs / total_pairs


def unique_chunk_rate(chunk_ids: list[str | None]) -> float:
    if not chunk_ids:
        return 1.0
    return len(set(chunk_ids)) / len(chunk_ids)


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    if not expected:
        return 1.0
    if not ranked:
        return 0.0
    top = set(ranked[:k])
    return len(expected & top) / len(expected)


def reciprocal_rank(expected: set[str], ranked: list[str]) -> float:
    for index, source in enumerate(ranked, start=1):
        if source in expected:
            return 1.0 / index
    return 0.0 if expected else 1.0


def substring_check(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    haystack = (text or "").lower()
    missing = [needle for needle in needles if needle.lower() not in haystack]
    return not missing, missing


def forbidden_source_hits(
    retrieved_sources: list[str],
    *,
    forbidden_sources: list[str] | None = None,
    forbidden_sources_prefix: str | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Sources in ``retrieved_sources`` that violate negative-case constraints."""
    hits: list[str] = []
    forbidden_set = set(forbidden_sources or [])
    prefix = (forbidden_sources_prefix or "").strip()
    scoped = retrieved_sources if top_n is None else retrieved_sources[: max(0, top_n)]
    for source in scoped:
        src = str(source or "")
        if src in forbidden_set:
            hits.append(src)
            continue
        if prefix and src.startswith(prefix):
            hits.append(src)
    return hits


def load_corpus(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema: {data.get('schema')!r}")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("corpus must contain a non-empty 'cases' list")
    return data


def corpus_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def evaluate_case(
    case: dict[str, Any],
    *,
    embed_fn,
    store: Any,
    config: LibraryEvalConfig,
) -> LibraryEvalResult:
    from mcp.rag_tool import rag_search

    case_id = str(case.get("id") or "")
    query = str(case.get("query") or "").strip()
    category = str(case.get("category") or "general")
    notes = str(case.get("notes") or "")
    expected_sources = [str(s) for s in (case.get("expected_sources") or [])]
    expect_contains = [str(s) for s in (case.get("expect_contains") or [])]
    expect_hits = case.get("expect_hits")
    forbidden_sources = [str(s) for s in (case.get("forbidden_sources") or [])]
    forbidden_sources_prefix = str(case.get("forbidden_sources_prefix") or "").strip() or None
    forbidden_in_top_n = case.get("forbidden_in_top_n")
    if forbidden_in_top_n is not None:
        forbidden_in_top_n = int(forbidden_in_top_n)

    query_vector = np.asarray(embed_fn(query), dtype=np.float32)
    result = rag_search(query, query_vector, store, top_k=config.top_k)
    sources = result.get("sources") or []

    retrieved_sources = [str(s.get("filename") or "") for s in sources]
    retrieved_text = "\n".join(str(s.get("content") or "") for s in sources)
    chunk_ids = [s.get("chunk_id") for s in sources]

    expected_set = set(expected_sources)
    hit_count = len(sources)
    rec = recall_at_k(expected_set, retrieved_sources, config.top_k)
    rr = reciprocal_rank(expected_set, retrieved_sources)
    sub_ok, missing = substring_check(retrieved_text, expect_contains)
    dup_rate = duplicate_pair_rate(
        [str(s.get("content") or "") for s in sources],
        threshold=config.duplicate_jaccard_threshold,
    )
    uniq_rate = unique_chunk_rate(chunk_ids)
    blocked_hits = forbidden_source_hits(
        retrieved_sources,
        forbidden_sources=forbidden_sources,
        forbidden_sources_prefix=forbidden_sources_prefix,
        top_n=forbidden_in_top_n,
    )

    failure_reason = "no_failure"
    success = True

    if forbidden_sources or forbidden_sources_prefix:
        if blocked_hits:
            success = False
            failure_reason = "forbidden_source_hit"
        elif expect_contains and not sub_ok:
            success = False
            failure_reason = "substring_miss"
    elif expect_hits is not None:
        if hit_count != int(expect_hits):
            success = False
            failure_reason = "unexpected_hit_count"
    elif not expected_set:
        if hit_count > 0:
            success = False
            failure_reason = "unexpected_hits"
    else:
        if rec <= 0.0:
            success = False
            failure_reason = "source_miss"
        elif expect_contains and not sub_ok:
            success = False
            failure_reason = "substring_miss"

    return LibraryEvalResult(
        case_id=case_id,
        query=query,
        category=category,
        expected_sources=expected_sources,
        retrieved_sources=retrieved_sources,
        hit_count=hit_count,
        recall_at_k=rec,
        reciprocal_rank=rr,
        substring_success=sub_ok,
        duplicate_pair_rate=round(dup_rate, 4),
        unique_chunk_rate=round(uniq_rate, 4),
        success=success,
        failure_reason=failure_reason,
        expect_contains=expect_contains,
        missing_substrings=missing,
        forbidden_source_hits=blocked_hits,
        notes=notes,
    )


def collect_index_stats(store: Any) -> dict[str, Any]:
    rows = store.export_all_rows()
    library_rows = [
        r for r in rows
        if str(r.get("source") or "").startswith(EVAL_LIBRARY_PREFIX)
    ]
    if not library_rows:
        return {
            "total_rows": 0,
            "library_rows": 0,
            "library_sources": 0,
            "avg_chunk_chars": 0.0,
            "avg_est_tokens": 0.0,
            "meta_json_coverage": 0.0,
        }

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in library_rows:
        source = str(row.get("source") or "")
        by_source.setdefault(source, []).append(row)

    char_lengths = [len(str(r.get("text") or "")) for r in library_rows]
    meta_count = sum(1 for r in library_rows if str(r.get("meta_json") or "").strip())

    return {
        "total_rows": len(rows),
        "library_rows": len(library_rows),
        "library_sources": len(by_source),
        "avg_chunk_chars": round(sum(char_lengths) / len(char_lengths), 1),
        "avg_est_tokens": round(sum(char_lengths) / len(char_lengths) / 4.0, 1),
        "meta_json_coverage": round(meta_count / len(library_rows), 3),
        "chunks_per_source": {
            source: len(items) for source, items in sorted(by_source.items())
        },
    }


def build_summary(
    results: list[LibraryEvalResult],
    *,
    corpus_path: Path,
    seed_summary: dict[str, Any] | None,
    index_stats: dict[str, Any],
    config: LibraryEvalConfig,
) -> dict[str, Any]:
    positive = [r for r in results if r.expected_sources]
    negative = [r for r in results if not r.expected_sources]

    def _mean(values: Iterable[float]) -> float:
        items = list(values)
        return round(sum(items) / len(items), 4) if items else 0.0

    summary = {
        "schema": RUN_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "corpus_fingerprint": corpus_fingerprint(corpus_path),
        "case_count": len(results),
        "success_count": sum(1 for r in results if r.success),
        "success_rate": round(sum(1 for r in results if r.success) / len(results), 4)
        if results
        else 0.0,
        "recall_at_k_mean": _mean(r.recall_at_k for r in positive),
        "mrr_mean": _mean(r.reciprocal_rank for r in positive),
        "substring_success_rate": _mean(1.0 if r.substring_success else 0.0 for r in positive),
        "duplicate_pair_rate_mean": _mean(r.duplicate_pair_rate for r in results),
        "unique_chunk_rate_mean": _mean(r.unique_chunk_rate for r in results),
        "negative_case_pass_rate": _mean(1.0 if r.success else 0.0 for r in negative),
        "top_k": config.top_k,
        "index_stats": index_stats,
        "seed_summary": seed_summary or {},
        "failures_by_reason": {},
    }

    for result in results:
        if result.success:
            continue
        summary["failures_by_reason"][result.failure_reason] = (
            summary["failures_by_reason"].get(result.failure_reason, 0) + 1
        )
    return summary


def format_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Library chunking eval summary",
        f"  cases: {summary.get('case_count')} | success: {summary.get('success_rate'):.1%}",
        f"  recall@{summary.get('top_k')} (positive): {summary.get('recall_at_k_mean'):.3f}",
        f"  MRR (positive): {summary.get('mrr_mean'):.3f}",
        f"  substring pass (positive): {summary.get('substring_success_rate'):.1%}",
        f"  duplicate pair rate (mean): {summary.get('duplicate_pair_rate_mean'):.3f}",
        f"  unique chunk rate (mean): {summary.get('unique_chunk_rate_mean'):.3f}",
    ]
    index_stats = summary.get("index_stats") or {}
    if index_stats.get("library_rows"):
        lines.extend([
            "  index:",
            f"    library rows: {index_stats.get('library_rows')}",
            f"    avg chunk chars: {index_stats.get('avg_chunk_chars')}",
            f"    avg est tokens: {index_stats.get('avg_est_tokens')}",
            f"    meta_json coverage: {index_stats.get('meta_json_coverage'):.1%}",
        ])
    seed = summary.get("seed_summary") or {}
    if seed and not seed.get("skipped"):
        lines.append(
            f"  seed: {seed.get('library_files')} files, "
            f"{seed.get('library_chunks')} chunks in {seed.get('ingest_elapsed_s')}s"
        )
    failures = summary.get("failures_by_reason") or {}
    if failures:
        lines.append(f"  failures: {failures}")
    return "\n".join(lines)


def write_csv(path: Path, results: list[LibraryEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["expected_sources"] = "|".join(result.expected_sources)
            row["retrieved_sources"] = "|".join(result.retrieved_sources)
            row["expect_contains"] = "|".join(result.expect_contains)
            row["missing_substrings"] = "|".join(result.missing_substrings)
            row["forbidden_source_hits"] = "|".join(result.forbidden_source_hits)
            writer.writerow(row)


def write_run_json(path: Path, summary: dict[str, Any], results: list[LibraryEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_runs(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    regressions: list[str] = []
    checks = [
        ("success_rate", "ge"),
        ("recall_at_k_mean", "ge"),
        ("mrr_mean", "ge"),
        ("substring_success_rate", "ge"),
        ("negative_case_pass_rate", "ge"),
    ]
    for key, direction in checks:
        cur = float(current.get(key) or 0.0)
        base = float(baseline.get(key) or 0.0)
        if direction == "ge" and cur + 1e-9 < base:
            regressions.append(f"{key} regressed: {cur:.4f} < baseline {base:.4f}")
    cur_dup = float(current.get("duplicate_pair_rate_mean") or 0.0)
    base_dup = float(baseline.get("duplicate_pair_rate_mean") or 0.0)
    if cur_dup > base_dup + 0.05:
        regressions.append(
            f"duplicate_pair_rate_mean worsened: {cur_dup:.4f} > baseline {base_dup:.4f} + 0.05"
        )
    return regressions
