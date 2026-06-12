"""
Shadow-mode route perturbation invariance harness.

Generates controlled paraphrases per corpus case and measures routing /
retrieval consistency. Does NOT modify routing logic or baseline eval results.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.router_evaluation import (
    RouterEvalCase,
    RouterEvalConfig,
    RouterEvalResult,
    evaluate_case,
    install_router_centroids,
    normalize_route,
)
from mcp.cognitive_router import CognitiveRouterV4

logger = logging.getLogger("Qube.RoutePerturbation")

PERTURBATION_SCHEMA = "qube.route_perturbation.v1"
MIN_VARIANTS = 3
MAX_VARIANTS = 6

_ENTITY_SWAPS: tuple[tuple[str, str], ...] = (
    ("python", "go"),
    ("java", "rust"),
    ("linux", "windows"),
    ("kubernetes", "docker"),
    ("tensorflow", "pytorch"),
    ("paris", "berlin"),
    ("luna", "max"),
    ("evelyn", "helen"),
    ("mark", "james"),
    ("sarah", "emily"),
)

_EXPLAIN_RE = re.compile(r"^explain\s+(.+?)[\?.!]*$", re.I)
_TELL_ME_RE = re.compile(r"^tell me about\s+(.+?)[\?.!]*$", re.I)
_WHAT_IS_RE = re.compile(r"^what is\s+(.+?)[\?.!]*$", re.I)
_HOW_DOES_RE = re.compile(r"^how does\s+(.+?)\s+work[\?.!]*$", re.I)
_SEARCH_RE = re.compile(r"\b(search|find|look)\b", re.I)
_WEB_RE = re.compile(r"\b(weather|news|online|google|web)\b", re.I)

_CHAT_SCORE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0-0.3", 0.0, 0.3),
    ("0.3-0.5", 0.3, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    ("0.7-1.0", 0.7, 1.01),
)

_MARGIN_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20+", 0.20, float("inf")),
)

_PERTURBATION_CATEGORIES: frozenset[str] = frozenset({
    "general_knowledge_retrieval_tempting",
    "memory_recall",
    "rag_retrieval",
    "follow_up",
    "ambiguous",
})


@dataclass
class PerturbVariant:
    variant_id: str
    perturbation_type: str
    text: str


@dataclass
class VariantRunResult:
    variant_id: str
    text: str
    perturbation_type: str
    route: str
    execution_route: str
    memory_hits: int
    rag_hits: int
    web_hits: int
    confidence_margin: float
    top_score: float
    chat_score: float
    second_best_score: float = 0.0
    recall_fusion_triggered: bool = False


@dataclass
class CasePerturbationReport:
    case_id: str
    base_prompt: str
    category: str
    base_route: str
    variants: list[VariantRunResult]
    route_consistency_score: float
    retrieval_consistency_score: float
    web_trigger_stability: float
    stability_label: str
    unique_routes: list[str]
    route_variance_pattern: str
    retrieval_variance_pattern: str
    confidence_margins: list[float]


@dataclass
class RoutePerturbationAnalysis:
    summary: dict[str, Any]
    cases: list[CasePerturbationReport] = field(default_factory=list)


def _bucket(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str:
    for label, low, high in buckets:
        if low <= value < high:
            return label
    return buckets[-1][0]


def _topic_from_prompt(prompt: str) -> Optional[str]:
    p = prompt.strip()
    for pat in (_EXPLAIN_RE, _TELL_ME_RE, _WHAT_IS_RE, _HOW_DOES_RE):
        m = pat.match(p)
        if m:
            return m.group(1).strip(" .?!")
    return None


def generate_perturbations(case: RouterEvalCase | dict[str, Any]) -> list[dict[str, Any]]:
    """
    Produce 3–6 deterministic semantic perturbations for a corpus case.

    Returns list of dicts with keys: variant_id, perturbation_type, text.
    """
    if isinstance(case, RouterEvalCase):
        case_id = case.id
        prompt = case.prompt
        history = case.history
    else:
        case_id = str(case.get("id") or "case")
        prompt = str(case.get("prompt") or "")
        history = tuple(case.get("history") or ())

    base = prompt.strip()
    if not base:
        return []

    variants: list[dict[str, Any]] = []
    seen: set[str] = {base.lower()}

    def _add(vid: str, ptype: str, text: str) -> None:
        t = text.strip()
        if not t or t.lower() in seen or len(variants) >= MAX_VARIANTS:
            return
        seen.add(t.lower())
        variants.append({
            "variant_id": f"{case_id}__{vid}",
            "perturbation_type": ptype,
            "text": t,
        })

    topic = _topic_from_prompt(base)

    if _EXPLAIN_RE.match(base) and topic:
        _add("para_how", "paraphrase", f"How does {topic} work?")
        _add("para_what", "paraphrase", f"What is {topic}?")
    elif _TELL_ME_RE.match(base) and topic:
        _add("para_what", "paraphrase", f"What is {topic}?")
        _add("para_explain", "paraphrase", f"Explain {topic}.")
    elif _WHAT_IS_RE.match(base) and topic:
        _add("para_explain", "paraphrase", f"Explain {topic}.")
        _add("para_tell", "paraphrase", f"Tell me about {topic}.")
    else:
        _add("para_explain", "paraphrase", f"Explain {base.rstrip('.?!')}.")
        _add("para_what", "paraphrase", f"What is {base.rstrip('.?!')}?")

    if topic:
        _add("expand", "expansion", f"Give a detailed explanation of {topic}.")
        _add("compress", "compression", f"Briefly define {topic}.")
    else:
        _add("expand", "expansion", f"Give a detailed explanation: {base}")
        _add("compress", "compression", f"Briefly: {base}")

    if history:
        _add("deixis_about", "deixis", "What about it?")
        _add("deixis_compare", "deixis", "How does that compare?")

    lowered = base.lower()
    for old, new in _ENTITY_SWAPS:
        if old in lowered:
            swapped = re.sub(re.escape(old), new, base, count=1, flags=re.I)
            _add(f"swap_{old}", "entity_swap", swapped)
            break

    words = base.split()
    if len(words) >= 4:
        last = words[-1].rstrip(".?!")
        if last.lower() not in ("it", "this", "that", "them") and last.isalpha():
            pronoun = "it" if last.lower() not in ("linux", "kubernetes") else "this"
            amb = " ".join(words[:-1]) + f" — what about {pronoun}?"
            _add("ambig_pronoun", "ambiguity", amb)

    if len(variants) < MIN_VARIANTS:
        _add("para_rephrase", "paraphrase", f"Can you help me understand: {base}")
        _add("para_rephrase2", "paraphrase", f"I'd like to know: {base.rstrip('.?!')}?")

    return variants[:MAX_VARIANTS]


def _embedder_adapter(embed_fn: Any) -> Any:
    class _Adapter:
        def embed_query(self, text: str):
            return embed_fn(text)

    return _Adapter()


def _make_router(embed_fn: Any, config: RouterEvalConfig) -> CognitiveRouterV4:
    router = CognitiveRouterV4()
    if config.install_centroids and embed_fn is not None:
        install_router_centroids(router, _embedder_adapter(embed_fn))
    return router


def run_router_on_variants(
    base_case: RouterEvalCase,
    variants: list[dict[str, Any]],
    *,
    embed_fn: Any,
    config: RouterEvalConfig,
    store: Any = None,
) -> list[VariantRunResult]:
    """Execute router harness on each perturbation variant."""
    out: list[VariantRunResult] = []
    router = _make_router(embed_fn, config)

    for var in variants:
        variant_case = RouterEvalCase(
            id=str(var["variant_id"]),
            prompt=str(var["text"]),
            expected_route=base_case.expected_route,
            category=base_case.category,
            notes=base_case.notes,
            history=base_case.history,
            flags=base_case.flags,
        )
        result = evaluate_case(
            variant_case,
            router=router,
            embed_fn=embed_fn,
            config=config,
            store=store,
            sidecar_client=None,
        )
        out.append(
            VariantRunResult(
                variant_id=str(var["variant_id"]),
                text=str(var["text"]),
                perturbation_type=str(var.get("perturbation_type") or ""),
                route=result.router_route,
                execution_route=result.execution_route_final,
                memory_hits=result.memory_hits,
                rag_hits=result.rag_hits,
                web_hits=result.web_hits,
                confidence_margin=result.confidence_margin,
                top_score=result.top_score,
                chat_score=result.chat_score,
                second_best_score=result.second_best_score,
                recall_fusion_triggered=result.recall_fusion_triggered,
            )
        )
    return out


def _route_consistency(variants: list[VariantRunResult]) -> float:
    if not variants:
        return 1.0
    routes = [normalize_route(v.execution_route) for v in variants]
    unique = len(set(routes))
    return 1.0 - (unique / len(routes))


def _retrieval_consistency(variants: list[VariantRunResult]) -> float:
    if not variants:
        return 1.0
    flags = [
        1 if (v.memory_hits + v.rag_hits + v.web_hits) > 0 else 0
        for v in variants
    ]
    if len(flags) < 2:
        return 1.0
    return 1.0 - statistics.pvariance(flags)


def _web_trigger_stability(variants: list[VariantRunResult]) -> float:
    if not variants:
        return 1.0
    web_flags = [normalize_route(v.execution_route) == "web" for v in variants]
    if all(web_flags) or not any(web_flags):
        return 1.0
    return 1.0 - (sum(web_flags) / len(web_flags))


def _stability_label(route_consistency: float) -> str:
    if route_consistency < 0.6:
        return "highly_unstable"
    if route_consistency < 0.85:
        return "moderately_unstable"
    return "stable"


def _variance_pattern(values: list[str]) -> str:
    unique = sorted(set(values))
    if len(unique) <= 1:
        return unique[0] if unique else "none"
    return " ↔ ".join(unique)


def _retrieval_pattern(variants: list[VariantRunResult]) -> str:
    parts = []
    for v in variants:
        hits = v.memory_hits + v.rag_hits + v.web_hits
        parts.append("hits" if hits > 0 else "no_hits")
    if len(set(parts)) == 1:
        return parts[0]
    return f"{parts.count('hits')}hits/{parts.count('no_hits')}miss"


def analyze_case_perturbation(
    base_case: RouterEvalCase,
    base_result: RouterEvalResult,
    variants: list[dict[str, Any]],
    *,
    embed_fn: Any,
    config: RouterEvalConfig,
    store: Any = None,
) -> CasePerturbationReport:
    runs = run_router_on_variants(
        base_case,
        variants,
        embed_fn=embed_fn,
        config=config,
        store=store,
    )
    route_cons = _route_consistency(runs)
    retr_cons = _retrieval_consistency(runs)
    web_stab = _web_trigger_stability(runs)
    routes = [normalize_route(v.execution_route) for v in runs]

    return CasePerturbationReport(
        case_id=base_case.id,
        base_prompt=base_case.prompt,
        category=base_case.category,
        base_route=base_result.execution_route_final,
        variants=runs,
        route_consistency_score=round(route_cons, 4),
        retrieval_consistency_score=round(retr_cons, 4),
        web_trigger_stability=round(web_stab, 4),
        stability_label=_stability_label(route_cons),
        unique_routes=sorted(set(routes)),
        route_variance_pattern=_variance_pattern(routes),
        retrieval_variance_pattern=_retrieval_pattern(runs),
        confidence_margins=[round(v.confidence_margin, 4) for v in runs],
    )


def _variance_by_route_type(
    case_reports: list[CasePerturbationReport],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for cr in case_reports:
        route = normalize_route(cr.base_route)
        b = buckets.setdefault(
            route,
            {"count": 0, "route_consistency_sum": 0.0, "unstable_count": 0},
        )
        b["count"] += 1
        b["route_consistency_sum"] += cr.route_consistency_score
        if cr.stability_label != "stable":
            b["unstable_count"] += 1
    for stats in buckets.values():
        n = stats["count"]
        stats["avg_route_consistency"] = stats["route_consistency_sum"] / n if n else 0.0
        stats["unstable_rate"] = stats["unstable_count"] / n if n else 0.0
    return buckets


def _instability_heatmap(
    case_reports: list[CasePerturbationReport],
    base_results: dict[str, RouterEvalResult],
) -> dict[str, dict[str, int]]:
    heat: dict[str, dict[str, int]] = {
        cs: {mb: 0 for mb, _, _ in _MARGIN_BUCKETS}
        for cs, _, _ in _CHAT_SCORE_BUCKETS
    }
    for cr in case_reports:
        if cr.stability_label == "stable":
            continue
        base = base_results.get(cr.case_id)
        if base is None:
            continue
        cs_bucket = _bucket(base.chat_score, _CHAT_SCORE_BUCKETS)
        margin_bucket = _bucket(base.confidence_margin, _MARGIN_BUCKETS)
        heat[cs_bucket][margin_bucket] = heat.get(cs_bucket, {}).get(margin_bucket, 0) + 1
    return heat


def analyze_route_perturbation(
    cases: list[RouterEvalCase],
    base_results: list[RouterEvalResult],
    *,
    embed_fn: Any,
    config: RouterEvalConfig,
    store: Any = None,
    run_id: str = "",
    cache_dir: Optional[Path] = None,
    corpus_fingerprint: str = "",
) -> RoutePerturbationAnalysis:
    """Run full perturbation invariance analysis (shadow mode)."""
    base_by_id = {r.case_id: r for r in base_results}
    perturbation_cache: dict[str, list[dict[str, Any]]] = {}
    case_reports: list[CasePerturbationReport] = []

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = corpus_fingerprint or "default"
        cache_path = cache_dir / f"perturbations_{fp}_{run_id or 'run'}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                perturbation_cache = cached.get("perturbations") or {}
                logger.info("Loaded perturbation cache: %s", cache_path)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Perturbation cache read failed: %s", exc)

    for case in cases:
        if case.id not in base_by_id:
            continue
        if case.id in perturbation_cache:
            variants = perturbation_cache[case.id]
        else:
            variants = generate_perturbations(case)
            perturbation_cache[case.id] = variants

        if len(variants) < MIN_VARIANTS:
            logger.debug("Skipping %s: only %d variants", case.id, len(variants))
            continue

        report = analyze_case_perturbation(
            case,
            base_by_id[case.id],
            variants,
            embed_fn=embed_fn,
            config=config,
            store=store,
        )
        case_reports.append(report)

    if cache_path is not None:
        cache_path.write_text(
            json.dumps(
                {
                    "schema": "qube.perturbation_cache.v1",
                    "run_id": run_id,
                    "corpus_fingerprint": corpus_fingerprint,
                    "perturbations": perturbation_cache,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    n = len(case_reports)
    route_scores = [c.route_consistency_score for c in case_reports]
    retr_scores = [c.retrieval_consistency_score for c in case_reports]
    stable = sum(1 for c in case_reports if c.stability_label == "stable")
    moderate = sum(1 for c in case_reports if c.stability_label == "moderately_unstable")
    highly = sum(1 for c in case_reports if c.stability_label == "highly_unstable")

    by_category: dict[str, dict[str, Any]] = {}
    for cat in _PERTURBATION_CATEGORIES:
        rows = [c for c in case_reports if c.category == cat]
        if not rows:
            continue
        by_category[cat] = {
            "count": len(rows),
            "avg_route_consistency": sum(r.route_consistency_score for r in rows) / len(rows),
            "avg_retrieval_consistency": sum(r.retrieval_consistency_score for r in rows) / len(rows),
            "unstable_rate": sum(1 for r in rows if r.stability_label != "stable") / len(rows),
            "highly_unstable_rate": sum(1 for r in rows if r.stability_label == "highly_unstable") / len(rows),
        }

    top_unstable = sorted(
        case_reports,
        key=lambda c: (c.route_consistency_score, c.retrieval_consistency_score),
    )[:10]

    summary = {
        "cases_analyzed": n,
        "avg_route_consistency": sum(route_scores) / n if n else 0.0,
        "avg_retrieval_consistency": sum(retr_scores) / n if n else 0.0,
        "stable_rate": stable / n if n else 0.0,
        "moderately_unstable_rate": moderate / n if n else 0.0,
        "highly_unstable_rate": highly / n if n else 0.0,
        "unstable_rate": (moderate + highly) / n if n else 0.0,
        "avg_web_trigger_stability": (
            sum(c.web_trigger_stability for c in case_reports) / n if n else 0.0
        ),
        "variance_by_route_type": _variance_by_route_type(case_reports),
        "by_category": by_category,
        "instability_heatmap": _instability_heatmap(case_reports, base_by_id),
        "top_unstable_cases": [
            {
                "case_id": c.case_id,
                "base_prompt": c.base_prompt,
                "category": c.category,
                "route_variance_pattern": c.route_variance_pattern,
                "retrieval_variance_pattern": c.retrieval_variance_pattern,
                "route_consistency_score": c.route_consistency_score,
                "retrieval_consistency_score": c.retrieval_consistency_score,
                "stability_label": c.stability_label,
                "confidence_margins": c.confidence_margins,
                "unique_routes": c.unique_routes,
            }
            for c in top_unstable
        ],
    }

    return RoutePerturbationAnalysis(summary=summary, cases=case_reports)


def export_perturbation_json(path: Path, analysis: RoutePerturbationAnalysis) -> None:
    payload = {
        "schema": PERTURBATION_SCHEMA,
        "summary": analysis.summary,
        "cases": [
            {
                **{k: v for k, v in asdict(cr).items() if k != "variants"},
                "variants": [asdict(v) for v in cr.variants],
            }
            for cr in analysis.cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
