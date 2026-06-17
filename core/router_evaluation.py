"""
Offline routing evaluation harness for CognitiveRouterV4.

Pure evaluation logic (no Qt). Used by ``tools/evaluate_router.py`` and unit tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from core.memory_filters import (
    detect_explicit_remember,
    detect_file_search_intent,
    detect_narrative_intent,
    should_apply_recall_fusion,
)
from core.rag_trigger_routing import (
    apply_custom_rag_trigger_route,
    matches_custom_rag_trigger,
)
from core.query_resolution_evaluation import (
    evaluate_resolution_expectations,
    expectations_from_flags,
)
from core.router_centroid_examples import (
    CHAT_INTENT_EXAMPLES,
    MEMORY_INTENT_EXAMPLES,
    RAG_INTENT_EXAMPLES,
    RECALL_INTENT_EXAMPLES,
    WEB_INTENT_EXAMPLES,
)
from mcp.cognitive_router import CognitiveRouterV4
from mcp.routing_debug import build_retrieval_outcome_snapshot

logger = logging.getLogger("Qube.RouterEval")

CORPUS_SCHEMA = "qube.router_corpus.v1"
RUN_SCHEMA = "qube.router_eval_run.v1"

_ROUTE_ALIASES: dict[str, str] = {
    "chat": "none",
    "internet": "web",
}

_VALID_ROUTES: frozenset[str] = frozenset(
    {"none", "memory", "rag", "web", "hybrid"}
)

_RETRIEVAL_ROUTES: frozenset[str] = frozenset(
    {"memory", "rag", "hybrid", "web", "internet"}
)

_ROUTE_FAMILIES: dict[str, str] = {
    "none": "CHAT",
    "memory": "RETRIEVAL",
    "rag": "RETRIEVAL",
    "hybrid": "RETRIEVAL",
    "web": "WEB",
}

_FAILURE_REASONS: frozenset[str] = frozenset({
    "router_miss",
    "override_changed_route",
    "recall_fusion_upgrade",
    "web_veto",
    "empty_retrieval",
    "relevance_gate_removed_results",
    "downgrade_to_none",
    "query_rewrite_rejected",
    "route_label_mismatch",
    "no_failure",
    "error",
})


def normalize_route(route: str) -> str:
    r = str(route or "none").strip().lower()
    return _ROUTE_ALIASES.get(r, r)


def route_family(route: str) -> str:
    """Map a route to CHAT / RETRIEVAL / WEB family."""
    return _ROUTE_FAMILIES.get(normalize_route(route), "CHAT")


def family_match(expected_route: str, actual_route: str) -> bool:
    return route_family(expected_route) == route_family(actual_route)


_CHAT_CALIBRATION_CATEGORIES: frozenset[str] = frozenset({
    "general_knowledge_retrieval_tempting",
    "ambiguous",
    "adversarial",
    "follow_up",
})

_MARGIN_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20+", 0.20, float("inf")),
)

_CHAT_GUARD_EPSILON = 0.05


def total_retrieval_hits(result: RouterEvalResult) -> int:
    return result.memory_hits + result.rag_hits + result.web_hits


def is_chat_labeled(result: RouterEvalResult) -> bool:
    return route_family(result.expected_route) == "CHAT"


def is_over_retrieval(result: RouterEvalResult) -> bool:
    """CHAT-labeled prompt ended in RETRIEVAL/WEB with non-zero hits."""
    return (
        route_family(result.expected_route) == "CHAT"
        and route_family(result.execution_route_final) in ("RETRIEVAL", "WEB")
        and total_retrieval_hits(result) > 0
    )


def is_under_retrieval(result: RouterEvalResult) -> bool:
    """RETRIEVAL/WEB expected but ended CHAT with zero hits."""
    return (
        route_family(result.expected_route) in ("RETRIEVAL", "WEB")
        and route_family(result.execution_route_final) == "CHAT"
        and total_retrieval_hits(result) == 0
    )


def infer_retrieval_type(result: RouterEvalResult) -> str:
    if result.web_hits > 0:
        return "web"
    if result.memory_hits > 0 and result.rag_hits > 0:
        return "hybrid"
    if result.memory_hits > 0:
        return "memory"
    if result.rag_hits > 0:
        return "rag"
    return "none"


def _margin_bucket(margin: float) -> str:
    for label, low, high in _MARGIN_BUCKETS:
        if low <= margin < high:
            return label
    return "0.20+"


def _detect_recall_fusion_triggered(
    *,
    prompt: str,
    decision: dict[str, Any],
    override_reason: str,
    execution_pre: str,
) -> bool:
    if bool(decision.get("recall_fusion")):
        return True
    if "recall_fusion" in (override_reason or ""):
        return True
    if (
        normalize_route(execution_pre) == "hybrid"
        and should_apply_recall_fusion(
            prompt.lower().strip(),
            decision=decision,
        )
    ):
        return True
    return False


@dataclass(frozen=True)
class RouterEvalCase:
    id: str
    prompt: str
    expected_route: str
    category: str
    notes: str = ""
    history: tuple[dict[str, str], ...] = ()
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouterEvalConfig:
    discourse_enabled: bool = True
    internet_enabled: bool = False
    internet_hybrid_auto: bool = False
    mcp_auto_enabled: bool = True
    mcp_rag_enabled: bool = True
    custom_rag_triggers: tuple[str, ...] = ()
    install_centroids: bool = True
    with_retrieval: bool = False
    with_sidecar_rewrite: bool = False
    with_web_fixtures: bool = False
    web_fixtures_dir: Optional[Path] = None


@dataclass
class RouterEvalResult:
    case_id: str
    prompt: str
    expected_route: str
    category: str
    notes: str
    router_route: str
    execution_route_pre_retrieval: str
    execution_route_final: str
    top_intent: str
    top_score: float
    chat_score: float
    confidence_margin: float
    memory_hits: int
    rag_hits: int
    web_hits: int
    downgrade_fired: bool
    rewrite_applied: bool
    router_match: bool
    execution_pre_match: bool
    execution_final_match: bool
    strict_success: bool = False
    family_success: bool = False
    failure_reason: str = "no_failure"
    rewrite_attempted: bool = False
    query_expansion_confidence: float = 0.0
    hybrid_extra_memory: int = 0
    hybrid_extra_rag: int = 0
    memory_candidates: int = 0
    rag_candidates: int = 0
    relevance_gate_dropped: bool = False
    memory_type: str = ""
    second_best_score: float = 0.0
    recall_fusion_triggered: bool = False
    over_retrieval: bool = False
    under_retrieval: bool = False
    retrieval_type: str = "none"
    inference_text: str = ""
    web_text: str = ""
    query_resolution_pass: Optional[bool] = None
    query_resolution_failed_checks: list[str] = field(default_factory=list)
    web_fixture_hits: int = 0
    stability_cluster_id: str = ""
    stability_cluster_size: int = 0
    is_oscillating_cluster: bool = False
    oscillation_reason: str = ""
    override_reason: str = ""
    error: str = ""


@dataclass
class RouterEvalSummary:
    total: int
    router_accuracy: float
    execution_pre_accuracy: float
    execution_final_accuracy: float
    strict_accuracy: float
    family_accuracy: float
    downgrade_count: int
    downgrade_rate: float
    rewrite_applied_count: int
    failure_causes: dict[str, int]
    rewrite_impact: dict[str, Any]
    memory_analysis: dict[str, Any]
    by_expected_route: dict[str, dict[str, Any]]
    by_category: dict[str, dict[str, Any]]
    confusion_matrix: dict[str, dict[str, int]]
    retrieval_hit_rates: dict[str, float]
    retrieval_calibration: dict[str, Any]
    errors: list[str]


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


def load_corpus(path: Path) -> tuple[dict[str, Any], list[RouterEvalCase]]:
    """Load JSON router corpus; raises ValueError on schema violations."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("corpus root must be a JSON object")

    schema = str(data.get("schema") or "")
    if schema and schema != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema: {schema!r} (expected {CORPUS_SCHEMA})")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("corpus must contain a non-empty 'cases' list")

    cases: list[RouterEvalCase] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{idx}] must be an object")
        case_id = str(raw.get("id") or f"case_{idx:03d}").strip()
        if not case_id:
            raise ValueError(f"cases[{idx}] missing id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"cases[{idx}] ({case_id}) missing prompt")

        expected = normalize_route(str(raw.get("expected_route") or ""))
        if expected not in _VALID_ROUTES:
            raise ValueError(
                f"cases[{idx}] ({case_id}) invalid expected_route: {expected!r}"
            )

        category = str(raw.get("category") or "uncategorized").strip()
        notes = str(raw.get("notes") or "").strip()
        history = _parse_history(raw.get("history"))
        flags = raw.get("flags") if isinstance(raw.get("flags"), dict) else {}

        cases.append(
            RouterEvalCase(
                id=case_id,
                prompt=prompt,
                expected_route=expected,
                category=category,
                notes=notes,
                history=history,
                flags=flags,
            )
        )

    return data, cases


def corpus_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def install_router_centroids(router: CognitiveRouterV4, embedder: Any) -> None:
    from workers.intent_router import build_centroid

    if router.recall_centroid is None:
        router.set_recall_centroid(build_centroid(embedder, list(RECALL_INTENT_EXAMPLES)))
    if router.chat_centroid is None:
        router.set_chat_centroid(build_centroid(embedder, list(CHAT_INTENT_EXAMPLES)))
    if router.memory_centroid is None:
        router.set_memory_centroid(build_centroid(embedder, list(MEMORY_INTENT_EXAMPLES)))
    if router.rag_centroid is None:
        router.set_rag_centroid(build_centroid(embedder, list(RAG_INTENT_EXAMPLES)))
    if router.web_centroid is None:
        router.set_web_centroid(build_centroid(embedder, list(WEB_INTENT_EXAMPLES)))


def _discourse_context(
    prompt: str,
    history: tuple[dict[str, str], ...],
    *,
    discourse_enabled: bool,
):
    """Return ``(follow_up, discourse_state, resolved_retrieval_query)``."""
    from core.query_resolution_evaluation import build_discourse_resolution

    return build_discourse_resolution(
        prompt,
        history,
        discourse_enabled=discourse_enabled,
        apply_inference_rewrite=True,
    )


def simulate_execution_route(
    *,
    prompt: str,
    decision: dict[str, Any],
    config: RouterEvalConfig,
    follow_up: Any = None,
    discourse_state: Any = None,
    case_flags: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """
    Apply LLMWorker-style post-router overrides.

    Returns ``(execution_route, override_reason)``.
    """
    flags = case_flags or {}
    clean_prompt = prompt.lower().strip()
    execution_route = str(decision.get("route") or "none").upper()
    override_reason = ""

    explicit_remember_active = bool(detect_explicit_remember(prompt))
    file_search_active = (
        not explicit_remember_active and detect_file_search_intent(prompt)
    )
    narrative_active = (
        not explicit_remember_active
        and not file_search_active
        and detect_narrative_intent(prompt)
    )
    scoped_library_active = file_search_active

    internet_enabled = bool(flags.get("internet_enabled", config.internet_enabled))
    internet_hybrid_auto = bool(
        flags.get("internet_hybrid_auto", config.internet_hybrid_auto)
    )
    discourse_enabled = bool(flags.get("discourse_enabled", config.discourse_enabled))
    mcp_auto = bool(flags.get("mcp_auto_enabled", config.mcp_auto_enabled))
    mcp_rag_enabled = bool(flags.get("mcp_rag_enabled", config.mcp_rag_enabled))
    custom_triggers = tuple(flags.get("custom_rag_triggers") or config.custom_rag_triggers)

    if explicit_remember_active:
        return "NONE", "explicit_remember"
    if file_search_active:
        return "RAG", "explicit_file_search"
    if narrative_active:
        return "MEMORY", "narrative_recap"

    if (
        not scoped_library_active
        and should_apply_recall_fusion(clean_prompt, decision=decision)
        and execution_route in ("NONE", "MEMORY", "RAG")
    ):
        execution_route = "HYBRID"
        decision["recall_fusion"] = True
        override_reason = "recall_fusion"

    if (
        discourse_enabled
        and follow_up is not None
        and getattr(follow_up, "active", False)
        and discourse_state is not None
        and getattr(discourse_state, "active_topic", None)
        and execution_route in ("MEMORY", "RAG", "HYBRID")
        and not narrative_active
        and not decision.get("recall_fusion")
    ):
        execution_route = "NONE"
        override_reason = "discourse_follow_up_downgrade"

    if mcp_auto and not scoped_library_active:
        force_rag_via_trigger = matches_custom_rag_trigger(clean_prompt, custom_triggers)
        if force_rag_via_trigger:
            execution_route, _ = apply_custom_rag_trigger_route(
                execution_route, matched=True
            )
            override_reason = override_reason or "custom_rag_trigger"
    else:
        force_rag_via_trigger = False

    from core.memory_filters import (
        detect_hard_explicit_web_request,
        library_lane_allowed,
        query_implies_live_web_intent,
    )

    library_bypass = library_lane_allowed(
        mcp_rag_enabled=mcp_rag_enabled,
        force_rag_via_trigger=force_rag_via_trigger,
        scoped_library_active=scoped_library_active,
    )
    library_blocked = not library_bypass
    if library_blocked and execution_route == "RAG":
        execution_route = "NONE"
        decision["rag_vetoed_tool_disabled"] = True
        override_reason = override_reason or "rag_veto_tool_disabled"
    elif library_blocked and execution_route == "HYBRID":
        execution_route = "MEMORY"
        decision["rag_vetoed_tool_disabled"] = True
        decision["rag_library_leg_skipped"] = True
        override_reason = override_reason or "rag_veto_tool_disabled"

    hard_web = detect_hard_explicit_web_request(clean_prompt)
    live_web = query_implies_live_web_intent(clean_prompt, decision=decision)
    manual_web = hard_web
    auto_web = internet_hybrid_auto and bool(decision.get("internet_enabled"))

    if not explicit_remember_active and not scoped_library_active:
        if manual_web or auto_web or (live_web and not hard_web):
            execution_route = "WEB"
            override_reason = override_reason or "web_trigger"

        if (
            execution_route == "WEB"
            and not manual_web
            and not auto_web
            and not internet_enabled
        ):
            execution_route = "NONE"
            decision["web_vetoed_tool_disabled"] = True
            override_reason = override_reason or "web_veto_tool_disabled"

        if discourse_enabled and follow_up is not None and execution_route == "WEB":
            from core.discourse_query import should_veto_ungrounded_web_follow_up

            if (
                should_veto_ungrounded_web_follow_up(follow_up, discourse_state)
                and not manual_web
            ):
                execution_route = "NONE"
                override_reason = override_reason or "discourse_veto_web"

    return execution_route, override_reason


def _memory_vector_candidates(store: Any, query_vector: np.ndarray) -> int:
    try:
        from mcp.memory_tool import _build_tier_where_clause

        where = _build_tier_where_clause(
            include_preference=True,
            include_knowledge=True,
            include_context=True,
        )
        rows = (
            store.table.search(query_vector)
            .where(where)
            .limit(10)
            .to_list()
        )
        return len(rows)
    except Exception:
        return 0


def _rag_vector_candidates(store: Any, query_vector: np.ndarray) -> int:
    try:
        rows = store.table.search(query_vector).limit(10).to_list()
        return len(rows)
    except Exception:
        return 0


def _run_retrieval(
    *,
    execution_route: str,
    retrieval_query: str,
    query_vector: np.ndarray | None,
    store: Any,
    web_query: str = "",
    web_fixture_id: str = "",
    fixtures_dir: Path | None = None,
    embed_fn: Any = None,
) -> dict[str, Any]:
    memory_hits = 0
    rag_hits = 0
    web_hits = 0
    memory_candidates = 0
    rag_candidates = 0
    route = execution_route.upper()

    if route in ("MEMORY", "HYBRID") and store is not None and query_vector is not None:
        memory_candidates = _memory_vector_candidates(store, query_vector)
        try:
            from mcp.memory_tool import memory_search

            mem = memory_search(
                retrieval_query,
                query_vector,
                store,
                include_preference=True,
                include_knowledge=True,
                include_context=True,
            )
            memory_hits = len(mem.get("memory_sources") or [])
        except Exception as exc:
            logger.debug("memory_search failed in eval: %s", exc)

    if route in ("RAG", "HYBRID") and store is not None and query_vector is not None:
        rag_candidates = _rag_vector_candidates(store, query_vector)
        try:
            from mcp.rag_tool import rag_search

            rag = rag_search(retrieval_query, query_vector, store)
            rag_hits = len(rag.get("sources") or [])
        except Exception as exc:
            logger.debug("rag_search failed in eval: %s", exc)

    fixture_id = (web_fixture_id or "").strip()
    if (
        fixture_id
        and route in ("WEB", "INTERNET", "HYBRID")
        and fixtures_dir is not None
    ):
        try:
            from core.query_resolution_evaluation import run_web_fixture_retrieval

            web_out = run_web_fixture_retrieval(
                web_query or retrieval_query,
                fixture_id,
                fixtures_dir=fixtures_dir,
                embed_fn=embed_fn,
            )
            web_hits = int(web_out.get("web_hits") or 0)
        except Exception as exc:
            logger.debug("web fixture retrieval failed in eval: %s", exc)

    relevance_gate_dropped = (
        (memory_candidates > 0 and memory_hits == 0 and route in ("MEMORY", "HYBRID"))
        or (rag_candidates > 0 and rag_hits == 0 and route in ("RAG", "HYBRID"))
    )

    return {
        "memory_hits": memory_hits,
        "rag_hits": rag_hits,
        "web_hits": web_hits,
        "memory_candidates": memory_candidates,
        "rag_candidates": rag_candidates,
        "relevance_gate_dropped": relevance_gate_dropped,
    }


def classify_failure_reason(
    *,
    strict_success: bool,
    family_success: bool,
    expected_route: str,
    router_route: str,
    execution_pre: str,
    execution_final: str,
    override_reason: str,
    downgrade_fired: bool,
    memory_hits: int,
    rag_hits: int,
    web_hits: int,
    memory_candidates: int,
    rag_candidates: int,
    relevance_gate_dropped: bool,
    rewrite_attempted: bool,
    rewrite_applied: bool,
    error: str,
) -> str:
    if error:
        return "error"
    if strict_success:
        return "no_failure"

    total_hits = memory_hits + rag_hits + web_hits
    pre_retrieval = execution_pre in _RETRIEVAL_ROUTES

    if downgrade_fired:
        if relevance_gate_dropped:
            return "relevance_gate_removed_results"
        return "downgrade_to_none"

    if pre_retrieval and total_hits == 0:
        if relevance_gate_dropped or memory_candidates > 0 or rag_candidates > 0:
            return "relevance_gate_removed_results"
        return "empty_retrieval"

    if "web_veto" in override_reason:
        return "web_veto"

    if rewrite_attempted and not rewrite_applied:
        return "query_rewrite_rejected"

    if family_success and not strict_success:
        if (
            "recall_fusion" in override_reason
            or normalize_route(execution_final) == "hybrid"
        ):
            return "recall_fusion_upgrade"
        return "route_label_mismatch"

    if override_reason and normalize_route(execution_pre) != router_route:
        return "override_changed_route"

    if normalize_route(execution_final) == "hybrid" and route_family(expected_route) == "RETRIEVAL":
        return "recall_fusion_upgrade"

    if router_route != expected_route and execution_final != expected_route:
        return "router_miss"

    return "router_miss"


def infer_memory_type(case: RouterEvalCase) -> str:
    explicit = str((case.flags or {}).get("memory_type") or "").strip().lower()
    if explicit:
        return explicit
    notes = (case.notes or "").lower()
    prompt = case.prompt.lower()
    if any(w in prompt for w in ("prefer", "favorite", "like", "allergic", "bedtime", "vegetarian")):
        return "preference"
    if any(w in prompt for w in ("brother", "wife", "mom", "dr.", "dog", "thesis")):
        return "relationship" if "brother" in prompt or "wife" in prompt or "mom" in prompt else "personal_fact"
    if any(w in prompt for w in ("meeting", "trip", "agreed", "notes")):
        return "episodic"
    if "relationship" in notes:
        return "relationship"
    if "preference" in notes:
        return "preference"
    return "unknown"


def evaluate_case(
    case: RouterEvalCase,
    *,
    router: CognitiveRouterV4,
    embed_fn: Any,
    config: RouterEvalConfig,
    store: Any = None,
    sidecar_client: Any = None,
) -> RouterEvalResult:
    expected = case.expected_route
    query_resolution_pass: Optional[bool] = None
    query_resolution_failed_checks: list[str] = []
    web_fixture_hits = 0
    inference_text = ""
    web_text = ""
    try:
        follow_up, discourse_state, resolved = _discourse_context(
            case.prompt,
            case.history,
            discourse_enabled=config.discourse_enabled,
        )
        routing_query = resolved.routing_text
        retrieval_query = resolved.retrieval_text
        inference_text = resolved.inference_text
        web_text = resolved.web_text

        qr_expect = expectations_from_flags(case.flags)
        if qr_expect is not None:
            query_resolution_failed_checks = evaluate_resolution_expectations(
                resolved, qr_expect
            )
            if (
                qr_expect.web_fixture_id
                and config.with_web_fixtures
            ):
                fixtures_dir = config.web_fixtures_dir
                if fixtures_dir is None:
                    from core.query_resolution_evaluation import DEFAULT_WEB_FIXTURES_DIR

                    fixtures_dir = DEFAULT_WEB_FIXTURES_DIR
                try:
                    from core.query_resolution_evaluation import run_web_fixture_retrieval

                    web_out = run_web_fixture_retrieval(
                        resolved.web_text,
                        qr_expect.web_fixture_id,
                        fixtures_dir=fixtures_dir,
                        embed_fn=embed_fn,
                    )
                    web_fixture_hits = int(web_out.get("web_hits") or 0)
                    if web_fixture_hits < qr_expect.min_web_hits:
                        query_resolution_failed_checks.append(
                            f"web_hits_below_min:{web_fixture_hits}<{qr_expect.min_web_hits}"
                        )
                except Exception as exc:
                    query_resolution_failed_checks.append(f"web_fixture_error:{exc}")
            query_resolution_pass = not query_resolution_failed_checks

        intent_vector = None
        if embed_fn is not None:
            intent_vector = embed_fn(routing_query)

        decision = router.route(routing_query, intent_vector=intent_vector)
        router_route = normalize_route(str(decision.get("route") or "none"))

        execution_pre, override_reason = simulate_execution_route(
            prompt=case.prompt,
            decision=decision,
            config=config,
            follow_up=follow_up,
            discourse_state=discourse_state,
            case_flags=case.flags,
        )
        execution_pre_norm = normalize_route(execution_pre)

        rewrite_attempted = False
        rewrite_applied = False
        expansion = None
        query_expansion_confidence = 0.0
        hybrid_extra_memory = 0
        hybrid_extra_rag = 0
        active_follow_up = bool(
            follow_up is not None and getattr(follow_up, "active", False)
        )

        if config.with_sidecar_rewrite and sidecar_client is not None and active_follow_up:
            rewrite_attempted = True
            from core.sidecar_query_rewrite import propose_query_expansion

            hist = [dict(h) for h in case.history]
            expansion = propose_query_expansion(
                case.prompt,
                follow_up,
                discourse_state,
                hist,
                sidecar_client,
                tentative_route=execution_pre_norm,
                retrieval_query=retrieval_query,
            )
            rewrite_applied = expansion is not None
            if expansion:
                query_expansion_confidence = float(expansion.confidence or 0.0)
                if expansion.recommended_target:
                    decision["sidecar_recommended_target"] = expansion.recommended_target

        memory_hits = rag_hits = web_hits = 0
        memory_candidates = rag_candidates = 0
        relevance_gate_dropped = False
        web_fixture_id_for_retrieval = ""
        fixtures_dir_for_retrieval: Path | None = None
        if config.with_web_fixtures:
            qr_for_retrieval = expectations_from_flags(case.flags)
            if qr_for_retrieval and qr_for_retrieval.web_fixture_id:
                web_fixture_id_for_retrieval = qr_for_retrieval.web_fixture_id
                fixtures_dir_for_retrieval = config.web_fixtures_dir
                if fixtures_dir_for_retrieval is None:
                    from core.query_resolution_evaluation import DEFAULT_WEB_FIXTURES_DIR

                    fixtures_dir_for_retrieval = DEFAULT_WEB_FIXTURES_DIR

        if config.with_retrieval and embed_fn is not None:
            qv = embed_fn(retrieval_query)
            retrieval = _run_retrieval(
                execution_route=execution_pre_norm,
                retrieval_query=retrieval_query,
                query_vector=qv,
                store=store,
                web_query=web_text,
                web_fixture_id=web_fixture_id_for_retrieval,
                fixtures_dir=fixtures_dir_for_retrieval,
                embed_fn=embed_fn,
            )
            memory_hits = int(retrieval["memory_hits"])
            rag_hits = int(retrieval["rag_hits"])
            web_hits = int(retrieval["web_hits"])
            memory_candidates = int(retrieval["memory_candidates"])
            rag_candidates = int(retrieval["rag_candidates"])
            relevance_gate_dropped = bool(retrieval["relevance_gate_dropped"])

            if rewrite_applied and expansion is not None and store is not None:
                expanded_query = str(expansion.expanded_query or "").strip()
                if expanded_query:
                    qv2 = embed_fn(expanded_query)
                    rewritten = _run_retrieval(
                        execution_route=execution_pre_norm,
                        retrieval_query=expanded_query,
                        query_vector=qv2,
                        store=store,
                    )
                    hybrid_extra_memory = max(
                        0, int(rewritten["memory_hits"]) - memory_hits
                    )
                    hybrid_extra_rag = max(0, int(rewritten["rag_hits"]) - rag_hits)
                    memory_hits = int(rewritten["memory_hits"])
                    rag_hits = int(rewritten["rag_hits"])
                    web_hits = int(rewritten["web_hits"])
                    memory_candidates = max(
                        memory_candidates, int(rewritten["memory_candidates"])
                    )
                    rag_candidates = max(
                        rag_candidates, int(rewritten["rag_candidates"])
                    )
                    relevance_gate_dropped = relevance_gate_dropped or bool(
                        rewritten["relevance_gate_dropped"]
                    )

        if web_hits > 0 and web_fixture_id_for_retrieval:
            web_fixture_hits = max(web_fixture_hits, web_hits)

        execution_final = execution_pre_norm
        downgrade_fired = False
        if (
            execution_pre_norm in _RETRIEVAL_ROUTES
            and (memory_hits + rag_hits + web_hits) == 0
        ):
            snapshot = build_retrieval_outcome_snapshot(
                decision=decision,
                execution_route_pre_downgrade=execution_pre_norm,
                execution_route_final="none",
                memory_hits=memory_hits,
                rag_hits=rag_hits,
                web_hits=web_hits,
            )
            if snapshot.get("downgrade_fired"):
                execution_final = "none"
                downgrade_fired = True

        top_intent = str(decision.get("top_intent") or "")
        top_score = float(decision.get("top_score") or 0.0)
        chat_score = float(decision.get("chat_score") or 0.0)
        confidence_margin = float(decision.get("confidence_margin") or 0.0)
        second_best_score = float(decision.get("second_best_score") or 0.0)

        recall_fusion_triggered = _detect_recall_fusion_triggered(
            prompt=case.prompt,
            decision=decision,
            override_reason=override_reason,
            execution_pre=execution_pre_norm,
        )

        strict_success = execution_final == expected
        fam_success = family_match(expected, execution_final)
        failure_reason = classify_failure_reason(
            strict_success=strict_success,
            family_success=fam_success,
            expected_route=expected,
            router_route=router_route,
            execution_pre=execution_pre_norm,
            execution_final=execution_final,
            override_reason=override_reason,
            downgrade_fired=downgrade_fired,
            memory_hits=memory_hits,
            rag_hits=rag_hits,
            web_hits=web_hits,
            memory_candidates=memory_candidates,
            rag_candidates=rag_candidates,
            relevance_gate_dropped=relevance_gate_dropped,
            rewrite_attempted=rewrite_attempted,
            rewrite_applied=rewrite_applied,
            error="",
        )
        memory_type = (
            infer_memory_type(case)
            if case.category == "memory_recall"
            else ""
        )

        provisional = RouterEvalResult(
            case_id=case.id,
            prompt=case.prompt,
            expected_route=expected,
            category=case.category,
            notes=case.notes,
            router_route=router_route,
            execution_route_pre_retrieval=execution_pre_norm,
            execution_route_final=execution_final,
            top_intent=top_intent,
            top_score=top_score,
            chat_score=chat_score,
            confidence_margin=confidence_margin,
            memory_hits=memory_hits,
            rag_hits=rag_hits,
            web_hits=web_hits,
            downgrade_fired=downgrade_fired,
            rewrite_applied=rewrite_applied,
            router_match=(router_route == expected),
            execution_pre_match=(execution_pre_norm == expected),
            execution_final_match=strict_success,
            strict_success=strict_success,
            family_success=fam_success,
            failure_reason=failure_reason,
            rewrite_attempted=rewrite_attempted,
            query_expansion_confidence=query_expansion_confidence,
            hybrid_extra_memory=hybrid_extra_memory,
            hybrid_extra_rag=hybrid_extra_rag,
            memory_candidates=memory_candidates,
            rag_candidates=rag_candidates,
            relevance_gate_dropped=relevance_gate_dropped,
            memory_type=memory_type,
            second_best_score=second_best_score,
            recall_fusion_triggered=recall_fusion_triggered,
        )
        over_ret = is_over_retrieval(provisional)
        under_ret = is_under_retrieval(provisional)
        ret_type = infer_retrieval_type(provisional) if over_ret else "none"

        return RouterEvalResult(
            case_id=case.id,
            prompt=case.prompt,
            expected_route=expected,
            category=case.category,
            notes=case.notes,
            router_route=router_route,
            execution_route_pre_retrieval=execution_pre_norm,
            execution_route_final=execution_final,
            top_intent=top_intent,
            top_score=top_score,
            chat_score=chat_score,
            confidence_margin=confidence_margin,
            memory_hits=memory_hits,
            rag_hits=rag_hits,
            web_hits=web_hits,
            downgrade_fired=downgrade_fired,
            rewrite_applied=rewrite_applied,
            router_match=(router_route == expected),
            execution_pre_match=(execution_pre_norm == expected),
            execution_final_match=strict_success,
            strict_success=strict_success,
            family_success=fam_success,
            failure_reason=failure_reason,
            rewrite_attempted=rewrite_attempted,
            query_expansion_confidence=query_expansion_confidence,
            hybrid_extra_memory=hybrid_extra_memory,
            hybrid_extra_rag=hybrid_extra_rag,
            memory_candidates=memory_candidates,
            rag_candidates=rag_candidates,
            relevance_gate_dropped=relevance_gate_dropped,
            memory_type=memory_type,
            second_best_score=second_best_score,
            recall_fusion_triggered=recall_fusion_triggered,
            over_retrieval=over_ret,
            under_retrieval=under_ret,
            retrieval_type=ret_type,
            inference_text=inference_text,
            web_text=web_text,
            query_resolution_pass=query_resolution_pass,
            query_resolution_failed_checks=query_resolution_failed_checks,
            web_fixture_hits=web_fixture_hits,
            override_reason=override_reason,
        )
    except Exception as exc:
        logger.exception("eval case %s failed", case.id)
        return RouterEvalResult(
            case_id=case.id,
            prompt=case.prompt,
            expected_route=expected,
            category=case.category,
            notes=case.notes,
            router_route="error",
            execution_route_pre_retrieval="error",
            execution_route_final="error",
            top_intent="",
            top_score=0.0,
            chat_score=0.0,
            confidence_margin=0.0,
            memory_hits=0,
            rag_hits=0,
            web_hits=0,
            downgrade_fired=False,
            rewrite_applied=False,
            router_match=False,
            execution_pre_match=False,
            execution_final_match=False,
            strict_success=False,
            family_success=False,
            failure_reason="error",
            error=str(exc),
        )


def _bucket_stats(results: Iterable[RouterEvalResult]) -> dict[str, Any]:
    rows = list(results)
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "router_accuracy": 0.0,
            "execution_pre_accuracy": 0.0,
            "execution_final_accuracy": 0.0,
            "strict_accuracy": 0.0,
            "family_accuracy": 0.0,
            "downgrade_count": 0,
            "downgrade_rate": 0.0,
        }
    router_ok = sum(1 for r in rows if r.router_match)
    pre_ok = sum(1 for r in rows if r.execution_pre_match)
    final_ok = sum(1 for r in rows if r.strict_success)
    family_ok = sum(1 for r in rows if r.family_success)
    downgrades = sum(1 for r in rows if r.downgrade_fired)
    retrieval_attempts = [
        r for r in rows if r.execution_route_pre_retrieval in _RETRIEVAL_ROUTES
    ]
    retrieval_with_hits = [
        r
        for r in retrieval_attempts
        if (r.memory_hits + r.rag_hits + r.web_hits) > 0
    ]
    return {
        "total": total,
        "router_accuracy": router_ok / total,
        "execution_pre_accuracy": pre_ok / total,
        "execution_final_accuracy": final_ok / total,
        "strict_accuracy": final_ok / total,
        "family_accuracy": family_ok / total,
        "downgrade_count": downgrades,
        "downgrade_rate": downgrades / total,
        "retrieval_attempts": len(retrieval_attempts),
        "retrieval_hit_rate": (
            len(retrieval_with_hits) / len(retrieval_attempts)
            if retrieval_attempts
            else 0.0
        ),
    }


def _failure_cause_counts(results: list[RouterEvalResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        reason = r.failure_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def _rewrite_impact_summary(results: list[RouterEvalResult]) -> dict[str, Any]:
    attempted = [r for r in results if r.rewrite_attempted]
    applied = [r for r in results if r.rewrite_applied]
    total = len(results)
    return {
        "attempted_count": len(attempted),
        "applied_count": len(applied),
        "attempt_rate": len(attempted) / total if total else 0.0,
        "acceptance_rate": len(applied) / len(attempted) if attempted else 0.0,
        "avg_extra_memory_hits": (
            sum(r.hybrid_extra_memory for r in applied) / len(applied)
            if applied
            else 0.0
        ),
        "avg_extra_rag_hits": (
            sum(r.hybrid_extra_rag for r in applied) / len(applied)
            if applied
            else 0.0
        ),
    }


def _retrieval_calibration_summary(results: list[RouterEvalResult]) -> dict[str, Any]:
    chat_labeled = [r for r in results if is_chat_labeled(r)]
    over_cases = [r for r in results if r.over_retrieval]
    under_cases = [r for r in results if r.under_retrieval]
    retrieval_expected = [
        r for r in results if route_family(r.expected_route) in ("RETRIEVAL", "WEB")
    ]

    over_rate = len(over_cases) / len(chat_labeled) if chat_labeled else 0.0
    under_rate = len(under_cases) / len(retrieval_expected) if retrieval_expected else 0.0

    over_by_category: dict[str, dict[str, Any]] = {}
    for cat in sorted(_CHAT_CALIBRATION_CATEGORIES):
        rows = [r for r in chat_labeled if r.category == cat]
        if not rows:
            continue
        cat_over = [r for r in rows if r.over_retrieval]
        over_by_category[cat] = {
            "chat_labeled_total": len(rows),
            "over_retrieval_count": len(cat_over),
            "over_retrieval_rate": len(cat_over) / len(rows),
        }

    correct_chat = [
        r for r in chat_labeled if route_family(r.execution_route_final) == "CHAT"
    ]
    over_chat_scores = [r.chat_score for r in over_cases]
    correct_chat_scores = [r.chat_score for r in correct_chat]

    margin_hist: dict[str, int] = {label: 0 for label, _, _ in _MARGIN_BUCKETS}
    for r in chat_labeled:
        bucket = _margin_bucket(r.confidence_margin)
        margin_hist[bucket] = margin_hist.get(bucket, 0) + 1

    over_with_fusion = sum(1 for r in over_cases if r.recall_fusion_triggered)
    fusion_share = over_with_fusion / len(over_cases) if over_cases else 0.0

    median_correct_chat = 0.0
    if correct_chat_scores:
        sorted_scores = sorted(correct_chat_scores)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2:
            median_correct_chat = sorted_scores[mid]
        else:
            median_correct_chat = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0

    suppression_candidates = sorted(
        [
            {
                "case_id": r.case_id,
                "prompt": r.prompt,
                "chat_score": round(r.chat_score, 4),
                "top_intent": r.top_intent,
                "route_taken": r.execution_route_final,
                "retrieval_type": r.retrieval_type,
                "retrieval_hits": total_retrieval_hits(r),
                "recall_fusion_triggered": r.recall_fusion_triggered,
                "confidence_margin": round(r.confidence_margin, 4),
            }
            for r in over_cases
        ],
        key=lambda x: (-x["chat_score"], x["case_id"]),
    )

    return {
        "chat_labeled_total": len(chat_labeled),
        "over_retrieval_count": len(over_cases),
        "over_retrieval_rate": over_rate,
        "retrieval_necessity_error_count": len(under_cases),
        "under_retrieval_rate": under_rate,
        "retrieval_expected_total": len(retrieval_expected),
        "recall_fusion_over_retrieval_count": over_with_fusion,
        "recall_fusion_over_retrieval_share": fusion_share,
        "avg_chat_score_correct_chat_cases": (
            sum(correct_chat_scores) / len(correct_chat_scores)
            if correct_chat_scores
            else 0.0
        ),
        "avg_chat_score_over_retrieval_cases": (
            sum(over_chat_scores) / len(over_chat_scores) if over_chat_scores else 0.0
        ),
        "median_chat_score_correct_chat_cases": median_correct_chat,
        "potential_chat_guard_threshold_candidate": max(
            0.0, median_correct_chat - _CHAT_GUARD_EPSILON
        ),
        "chat_margin_histogram": margin_hist,
        "over_retrieval_by_category": over_by_category,
        "retrieval_suppression_candidates": suppression_candidates,
    }


def _memory_analysis_summary(results: list[RouterEvalResult]) -> dict[str, Any]:
    mem_rows = [r for r in results if r.category == "memory_recall"]
    if not mem_rows:
        return {"total": 0}

    with_hits = [r for r in mem_rows if r.memory_hits > 0]
    by_type: dict[str, dict[str, Any]] = {}
    for r in mem_rows:
        mtype = r.memory_type or "unknown"
        bucket = by_type.setdefault(
            mtype,
            {"total": 0, "hits": 0, "misses": 0, "strict_success": 0, "family_success": 0},
        )
        bucket["total"] += 1
        if r.memory_hits > 0:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1
        if r.strict_success:
            bucket["strict_success"] += 1
        if r.family_success:
            bucket["family_success"] += 1

    for stats in by_type.values():
        stats["hit_rate"] = stats["hits"] / stats["total"] if stats["total"] else 0.0

    return {
        "total": len(mem_rows),
        "with_hits": len(with_hits),
        "without_hits": len(mem_rows) - len(with_hits),
        "strict_success": sum(1 for r in mem_rows if r.strict_success),
        "family_success": sum(1 for r in mem_rows if r.family_success),
        "by_memory_type": by_type,
    }


def build_summary(results: list[RouterEvalResult]) -> RouterEvalSummary:
    by_expected: dict[str, list[RouterEvalResult]] = {}
    by_category: dict[str, list[RouterEvalResult]] = {}
    for r in results:
        by_expected.setdefault(r.expected_route, []).append(r)
        by_category.setdefault(r.category, []).append(r)

    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        exp = r.expected_route
        act = r.execution_route_final
        confusion.setdefault(exp, {})
        confusion[exp][act] = confusion[exp].get(act, 0) + 1

    retrieval_rates: dict[str, float] = {}
    for cat, rows in by_category.items():
        attempts = [x for x in rows if x.execution_route_pre_retrieval in _RETRIEVAL_ROUTES]
        if not attempts:
            retrieval_rates[cat] = 0.0
            continue
        with_hits = sum(
            1 for x in attempts if (x.memory_hits + x.rag_hits + x.web_hits) > 0
        )
        retrieval_rates[cat] = with_hits / len(attempts)

    overall = _bucket_stats(results)
    errors = [f"{r.case_id}: {r.error}" for r in results if r.error]

    return RouterEvalSummary(
        total=overall["total"],
        router_accuracy=overall["router_accuracy"],
        execution_pre_accuracy=overall["execution_pre_accuracy"],
        execution_final_accuracy=overall["execution_final_accuracy"],
        strict_accuracy=overall["strict_accuracy"],
        family_accuracy=overall["family_accuracy"],
        downgrade_count=overall["downgrade_count"],
        downgrade_rate=overall["downgrade_rate"],
        rewrite_applied_count=sum(1 for r in results if r.rewrite_applied),
        failure_causes=_failure_cause_counts(results),
        rewrite_impact=_rewrite_impact_summary(results),
        memory_analysis=_memory_analysis_summary(results),
        retrieval_calibration=_retrieval_calibration_summary(results),
        by_expected_route={
            k: _bucket_stats(v) for k, v in sorted(by_expected.items())
        },
        by_category={k: _bucket_stats(v) for k, v in sorted(by_category.items())},
        confusion_matrix=confusion,
        retrieval_hit_rates=retrieval_rates,
        errors=errors,
    )


def results_to_csv_rows(results: list[RouterEvalResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


def write_csv(path: Path, results: list[RouterEvalResult]) -> None:
    rows = results_to_csv_rows(results)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_json(
    path: Path,
    *,
    corpus_path: Path,
    corpus_meta: dict[str, Any],
    config: RouterEvalConfig,
    results: list[RouterEvalResult],
    summary: RouterEvalSummary,
    run_id: str,
    notes: str = "",
    routing_stability: Optional[dict[str, Any]] = None,
    route_perturbation: Optional[dict[str, Any]] = None,
    routing_hysteresis: Optional[dict[str, Any]] = None,
    routing_canonicalization: Optional[dict[str, Any]] = None,
    retrieval_propensity: Optional[dict[str, Any]] = None,
    continuous_pilot_routing: Optional[dict[str, Any]] = None,
    continuous_arch_validation: Optional[dict[str, Any]] = None,
    shadow_retrieval_policy: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "corpus_fingerprint": corpus_fingerprint(corpus_path),
        "corpus_version": corpus_meta.get("version"),
        "corpus_description": corpus_meta.get("description", ""),
        "notes": notes,
        "config": asdict(config),
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    if routing_stability is not None:
        payload["routing_stability"] = routing_stability
    if route_perturbation is not None:
        payload["route_perturbation"] = route_perturbation
    if routing_hysteresis is not None:
        payload["routing_hysteresis"] = routing_hysteresis
    if routing_canonicalization is not None:
        payload["routing_canonicalization"] = routing_canonicalization
    if retrieval_propensity is not None:
        payload["retrieval_propensity"] = retrieval_propensity
    if continuous_pilot_routing is not None:
        payload["continuous_pilot_routing"] = continuous_pilot_routing
    if continuous_arch_validation is not None:
        payload["continuous_arch_validation"] = continuous_arch_validation
    if shadow_retrieval_policy is not None:
        payload["shadow_retrieval_policy"] = shadow_retrieval_policy
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("run file must be a JSON object")
    return data


def compare_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    metric: str = "execution_final_accuracy",
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Compare summary metrics; ``regressed`` when current < baseline - min_delta."""
    b_sum = baseline.get("summary") if isinstance(baseline.get("summary"), dict) else {}
    c_sum = current.get("summary") if isinstance(current.get("summary"), dict) else {}

    b_val = float(b_sum.get(metric) or 0.0)
    c_val = float(c_sum.get(metric) or 0.0)
    delta = c_val - b_val
    regressed = delta < -abs(min_delta)

    new_failures: list[dict[str, str]] = []
    fixed_cases: list[dict[str, str]] = []
    b_results = {
        str(r.get("case_id")): r
        for r in (baseline.get("results") or [])
        if isinstance(r, dict)
    }
    c_results = {
        str(r.get("case_id")): r
        for r in (current.get("results") or [])
        if isinstance(r, dict)
    }
    for cid, cr in c_results.items():
        br = b_results.get(cid)
        if not br:
            continue
        base_ok = bool(br.get("execution_final_match"))
        cur_ok = bool(cr.get("execution_final_match"))
        row = {
            "case_id": cid,
            "baseline_final": str(br.get("execution_route_final")),
            "current_final": str(cr.get("execution_route_final")),
            "expected": str(cr.get("expected_route")),
        }
        if base_ok and not cur_ok:
            new_failures.append(row)
        elif not base_ok and cur_ok:
            fixed_cases.append(row)

    return {
        "metric": metric,
        "baseline": b_val,
        "current": c_val,
        "delta": delta,
        "regressed": regressed,
        "new_failures": new_failures,
        "fixed_cases": fixed_cases,
    }


def format_summary_text(summary: RouterEvalSummary) -> str:
    rc = summary.retrieval_calibration or {}
    lines = [
        f"Total cases: {summary.total}",
        f"Strict accuracy (final): {summary.strict_accuracy:.1%}",
        f"Family accuracy (final): {summary.family_accuracy:.1%}",
        f"Router accuracy: {summary.router_accuracy:.1%}",
        f"Execution pre-retrieval accuracy: {summary.execution_pre_accuracy:.1%}",
        f"Downgrades: {summary.downgrade_count} ({summary.downgrade_rate:.1%} of cases)",
        f"Sidecar rewrites applied: {summary.rewrite_applied_count}",
        (
            "Over-retrieval rate (CHAT leakage into retrieval): "
            f"{rc.get('over_retrieval_rate', 0):.1%} "
            f"({rc.get('over_retrieval_count', 0)}/{rc.get('chat_labeled_total', 0)})"
        ),
        (
            "Under-retrieval rate (missed retrieval): "
            f"{rc.get('under_retrieval_rate', 0):.1%} "
            f"({rc.get('retrieval_necessity_error_count', 0)}/"
            f"{rc.get('retrieval_expected_total', 0)})"
        ),
        "",
        "Failure causes:",
    ]
    over_by_cat = rc.get("over_retrieval_by_category") or {}
    if over_by_cat:
        lines.append("")
        lines.append("Over-retrieval by category:")
        for cat, stats in sorted(over_by_cat.items()):
            lines.append(
                f"  {cat}: {stats.get('over_retrieval_rate', 0):.1%} "
                f"({stats.get('over_retrieval_count', 0)}/"
                f"{stats.get('chat_labeled_total', 0)})"
            )
    for reason, count in (summary.failure_causes or {}).items():
        lines.append(f"  {reason}: {count}")
    lines.append("")
    lines.append("By category (strict / family):")
    for cat, stats in summary.by_category.items():
        lines.append(
            f"  {cat}: strict={stats['strict_accuracy']:.1%} "
            f"family={stats['family_accuracy']:.1%} "
            f"(n={stats['total']}, downgrades={stats['downgrade_count']})"
        )
    lines.append("")
    lines.append("By expected route (strict accuracy):")
    for route, stats in summary.by_expected_route.items():
        lines.append(
            f"  {route}: {stats['strict_accuracy']:.1%} (n={stats['total']})"
        )
    ri = summary.rewrite_impact or {}
    if ri.get("attempted_count"):
        lines.append("")
        lines.append("Rewrite impact:")
        lines.append(
            f"  attempt_rate={ri.get('attempt_rate', 0):.1%} "
            f"acceptance_rate={ri.get('acceptance_rate', 0):.1%}"
        )
        lines.append(
            f"  avg_extra_memory={ri.get('avg_extra_memory_hits', 0):.2f} "
            f"avg_extra_rag={ri.get('avg_extra_rag_hits', 0):.2f}"
        )
    ma = summary.memory_analysis or {}
    if ma.get("total"):
        lines.append("")
        lines.append(
            f"Memory recall: {ma.get('with_hits', 0)}/{ma.get('total', 0)} with hits, "
            f"strict={ma.get('strict_success', 0)} family={ma.get('family_success', 0)}"
        )
    lines.append("")
    lines.append("Confusion matrix (expected -> final):")
    for exp, acts in summary.confusion_matrix.items():
        act_str = ", ".join(f"{k}={v}" for k, v in sorted(acts.items()))
        lines.append(f"  {exp}: {act_str}")
    if summary.errors:
        lines.append("")
        lines.append(f"Errors: {len(summary.errors)}")
    return "\n".join(lines)
