"""
Post-hoc routing stability analysis (shadow mode).

Groups similar evaluation prompts into clusters and measures route consistency.
Does NOT influence routing decisions.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from core.router_evaluation import RouterEvalResult, normalize_route

logger = logging.getLogger("Qube.RoutingStability")

DEFAULT_SIMILARITY_THRESHOLD = 0.85

_INSTABILITY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.01),
)

_ENTROPY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0", 0.0, 0.001),
    ("0.0-0.5", 0.001, 0.5),
    ("0.5-1.0", 0.5, 1.0),
    ("1.0-1.5", 1.0, 1.5),
    ("1.5+", 1.5, float("inf")),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ClusterStats:
    cluster_id: str
    cases: list[str]
    routes: dict[str, int]
    dominant_route: str
    instability_score: float
    entropy: float
    is_oscillating: bool
    oscillation_reason: str = ""
    case_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingStabilityAnalysis:
    clusters: list[ClusterStats]
    summary: dict[str, Any]
    similarity_method: str


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def token_jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _cosine_similarity(va: np.ndarray, vb: np.ndarray) -> float:
    a = np.asarray(va, dtype=np.float32).flatten()
    b = np.asarray(vb, dtype=np.float32).flatten()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _bucket_value(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str:
    for label, low, high in buckets:
        if low <= value < high:
            return label
    return buckets[-1][0]


def _shannon_entropy(route_counts: dict[str, int]) -> float:
    total = sum(route_counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in route_counts.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_clusters(
    results: list[RouterEvalResult],
    *,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[dict[int, list[int]], str]:
    """
    Group case indices by semantic/token similarity (union-find).

    Returns ``(clusters_map, similarity_method)`` where keys are cluster
    representative indices and values are member index lists.
    """
    n = len(results)
    if n == 0:
        return {}, "none"

    vectors: list[Optional[np.ndarray]] = [None] * n
    method = "token_jaccard"

    if embed_fn is not None:
        try:
            for i, r in enumerate(results):
                vectors[i] = np.asarray(embed_fn(r.prompt), dtype=np.float32)
            method = "embedding_cosine"
        except Exception as exc:
            logger.warning("Embedding clustering unavailable (%s); using token Jaccard", exc)
            vectors = [None] * n
            method = "token_jaccard"

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if vectors[i] is not None and vectors[j] is not None:
                sim = _cosine_similarity(vectors[i], vectors[j])
            else:
                sim = token_jaccard_similarity(results[i].prompt, results[j].prompt)
            if sim >= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    return groups, method


def compute_cluster_metrics(
    cluster_id: str,
    members: list[RouterEvalResult],
) -> ClusterStats:
    route_counts: dict[str, int] = {}
    case_details: list[dict[str, Any]] = []
    for r in members:
        route = normalize_route(r.execution_route_final)
        route_counts[route] = route_counts.get(route, 0) + 1
        case_details.append({
            "case_id": r.case_id,
            "prompt": r.prompt,
            "category": r.category,
            "execution_route_final": route,
            "confidence_margin": round(r.confidence_margin, 4),
            "recall_fusion_triggered": r.recall_fusion_triggered,
        })

    total = len(members)
    dominant_route = max(route_counts, key=route_counts.get)
    dominant_count = route_counts[dominant_route]
    instability = 1.0 - (dominant_count / total) if total else 0.0
    entropy = _shannon_entropy(route_counts)

    is_osc, reason = detect_oscillation(members, route_counts)

    return ClusterStats(
        cluster_id=cluster_id,
        cases=[r.case_id for r in members],
        routes=route_counts,
        dominant_route=dominant_route,
        instability_score=round(instability, 4),
        entropy=round(entropy, 4),
        is_oscillating=is_osc,
        oscillation_reason=reason,
        case_details=case_details,
    )


def detect_oscillation(
    members: list[RouterEvalResult],
    route_counts: Optional[dict[str, int]] = None,
) -> tuple[bool, str]:
    """
    Flag clusters with ≥2 distinct routes and confidence_margin > 0.10 somewhere.
    """
    if route_counts is None:
        route_counts = {}
        for r in members:
            route = normalize_route(r.execution_route_final)
            route_counts[route] = route_counts.get(route, 0) + 1

    distinct_routes = set(route_counts.keys())
    if len(distinct_routes) < 2:
        return False, ""

    if not any(r.confidence_margin > 0.10 for r in members):
        return False, ""

    routes = distinct_routes
    if "hybrid" in routes and "none" in routes:
        return True, "hybrid_vs_none_flip"

    fusion_flags = {r.recall_fusion_triggered for r in members}
    if len(fusion_flags) > 1 and len(routes) >= 2:
        return True, "recall_fusion_inconsistent"

    fusion_routes = {
        normalize_route(r.execution_route_final)
        for r in members
        if r.recall_fusion_triggered
    }
    non_fusion_routes = {
        normalize_route(r.execution_route_final)
        for r in members
        if not r.recall_fusion_triggered
    }
    if fusion_routes and non_fusion_routes and fusion_routes != non_fusion_routes:
        return True, "recall_fusion_inconsistent"

    return True, "same_intent_different_routes"


def analyze_routing_stability(
    results: list[RouterEvalResult],
    *,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RoutingStabilityAnalysis:
    """Run full stability analysis; returns annotated results via ``apply_labels``."""
    groups, method = build_clusters(results, embed_fn=embed_fn, threshold=threshold)

    sorted_roots = sorted(
        groups.keys(),
        key=lambda root: (len(groups[root]), results[min(groups[root])].case_id),
        reverse=True,
    )

    clusters: list[ClusterStats] = []

    for seq, root in enumerate(sorted_roots):
        member_idxs = sorted(groups[root], key=lambda i: results[i].case_id)
        members = [results[i] for i in member_idxs]
        cluster_id = f"stab_{seq:04d}"
        stats = compute_cluster_metrics(cluster_id, members)
        clusters.append(stats)

    oscillating = [c for c in clusters if c.is_oscillating]
    entropies = [c.entropy for c in clusters]
    instabilities = [c.instability_score for c in clusters]

    instability_hist = {label: 0 for label, _, _ in _INSTABILITY_BUCKETS}
    entropy_hist = {label: 0 for label, _, _ in _ENTROPY_BUCKETS}
    for c in clusters:
        instability_hist[_bucket_value(c.instability_score, _INSTABILITY_BUCKETS)] += 1
        entropy_hist[_bucket_value(c.entropy, _ENTROPY_BUCKETS)] += 1

    top_unstable = sorted(
        clusters,
        key=lambda c: (c.instability_score, c.entropy, -len(c.cases)),
        reverse=True,
    )[:10]

    summary = {
        "total_clusters": len(clusters),
        "oscillating_clusters": len(oscillating),
        "oscillation_rate": len(oscillating) / len(clusters) if clusters else 0.0,
        "avg_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "avg_instability_score": (
            sum(instabilities) / len(instabilities) if instabilities else 0.0
        ),
        "similarity_method": method,
        "similarity_threshold": threshold,
        "instability_histogram": instability_hist,
        "entropy_histogram": entropy_hist,
        "max_instability_clusters": [
            {
                "cluster_id": c.cluster_id,
                "cases": c.cases,
                "routes": c.routes,
                "dominant_route": c.dominant_route,
                "instability_score": c.instability_score,
                "entropy": c.entropy,
                "is_oscillating": c.is_oscillating,
                "oscillation_reason": c.oscillation_reason,
            }
            for c in top_unstable
        ],
    }

    return RoutingStabilityAnalysis(
        clusters=clusters,
        summary=summary,
        similarity_method=method,
    )


def annotate_results_with_stability(
    results: list[RouterEvalResult],
    analysis: RoutingStabilityAnalysis,
) -> list[RouterEvalResult]:
    case_to_cluster = {cid: c for c in analysis.clusters for cid in c.cases}
    out: list[RouterEvalResult] = []
    for r in results:
        cluster = case_to_cluster.get(r.case_id)
        if cluster is None:
            out.append(r)
            continue
        out.append(
            replace(
                r,
                stability_cluster_id=cluster.cluster_id,
                stability_cluster_size=len(cluster.cases),
                is_oscillating_cluster=cluster.is_oscillating,
                oscillation_reason=cluster.oscillation_reason,
            )
        )
    return out


def export_stability_clusters_json(
    path: Path,
    analysis: RoutingStabilityAnalysis,
) -> None:
    payload = {
        "schema": "qube.routing_stability_clusters.v1",
        "summary": analysis.summary,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "cases": c.cases,
                "routes": c.routes,
                "dominant_route": c.dominant_route,
                "instability_score": c.instability_score,
                "entropy": c.entropy,
                "is_oscillating": c.is_oscillating,
                "oscillation_reason": c.oscillation_reason,
                "case_details": c.case_details,
            }
            for c in analysis.clusters
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
