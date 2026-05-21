"""Bounded in-memory routing debug records — observability only, no router logic."""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from core.engine_input_trace import (
    EngineInputTracer,
    engine_input_trace_enabled,
    engine_input_trace_to_public_dict,
)
from mcp.cognitive_router import AMBIGUITY_MARGIN, MIN_CONFIDENCE_FLOOR

MAX_RECORDS: int = 100
ROUTING_DEBUG_SCHEMA_VERSION: int = 1


@dataclass
class RoutingDebugRecord:
    timestamp: float
    session_id: Optional[str]
    turn_id: Optional[int]
    query: str
    route: str
    route_pre_policy: Optional[str]
    strategy: str
    trace_level: str
    top_intent: Optional[str]
    top_score: Optional[float]
    summary: str
    trace: dict[str, Any]
    decision: dict[str, Any] = field(repr=False)


def routing_debug_log_enabled() -> bool:
    return str(os.getenv("QUBE_ROUTING_DEBUG_LOG", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def routing_debug_log_verbose() -> bool:
    return str(os.getenv("QUBE_ROUTING_DEBUG_LOG_VERBOSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def routing_debug_log_redact_query() -> bool:
    return str(os.getenv("QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _redact_query(query: str) -> str:
    q = str(query or "")
    digest = hashlib.sha256(q.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"[redacted sha256:{digest}]"


def serialize_record_for_log(
    record: RoutingDebugRecord,
    *,
    verbose: bool = False,
    redact_query: bool = False,
) -> dict[str, Any]:
    """
    Stable JSONL payload for persisted routing-debug telemetry.
    """
    trace = _safe_dict(record.trace)
    confidence = _safe_dict(trace.get("confidence"))
    tier3 = _safe_dict(trace.get("tier3"))
    lane_bias = _safe_dict(tier3.get("lane_bias"))
    tier56 = _safe_dict(trace.get("tier5_6"))
    query = _redact_query(record.query) if redact_query else record.query

    payload: dict[str, Any] = {
        "schema_version": ROUTING_DEBUG_SCHEMA_VERSION,
        "ts": float(record.timestamp),
        "session_id": record.session_id,
        "turn_id": record.turn_id,
        "query": query,
        "route": record.route,
        "route_pre_policy": record.route_pre_policy,
        "strategy": record.strategy,
        "trace_level": record.trace_level,
        "top_intent": record.top_intent,
        "top_score": record.top_score,
        "summary": record.summary,
        "tier2_active": bool(confidence.get("tier2_active", False)),
        "tier3_band_active": bool(tier3.get("band_active", False)),
        "tier3_lane_bias": {
            "memory": float(lane_bias.get("memory") or 0.0),
            "rag": float(lane_bias.get("rag") or 0.0),
            "web": float(lane_bias.get("web") or 0.0),
        },
        "tier5_policy": tier56.get("policy"),
        "tier5_reason": tier56.get("policy_reason"),
        "tier6_conflicts": _safe_list(tier56.get("conflicts")),
        "tier6_interpretation": tier56.get("interpretation"),
    }

    model_router = trace.get("model_router")
    if isinstance(model_router, dict):
        payload["model_router"] = copy.deepcopy(model_router)
    chat_contract = trace.get("chat_contract")
    if isinstance(chat_contract, dict):
        payload["chat_contract"] = copy.deepcopy(chat_contract)
    engine_input = trace.get("engine_input_trace")
    if isinstance(engine_input, dict):
        payload["engine_input_trace"] = copy.deepcopy(engine_input)

    if verbose:
        payload["trace"] = copy.deepcopy(record.trace)
        payload["decision"] = copy.deepcopy(record.decision)

    return payload


class RoutingDebugBuffer:
    """Thread-safe bounded ring buffer for routing debug records."""

    def __init__(self, maxlen: int = MAX_RECORDS) -> None:
        self._deque: deque[RoutingDebugRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, record: RoutingDebugRecord) -> None:
        with self._lock:
            self._deque.append(record)

    def snapshot(self) -> list[RoutingDebugRecord]:
        """Chronological order: oldest first, newest last."""
        with self._lock:
            return list(self._deque)

    def latest(self) -> Optional[RoutingDebugRecord]:
        with self._lock:
            return self._deque[-1] if self._deque else None

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()

    def merge_model_router_into_latest(
        self, model_router: Optional[dict[str, Any]]
    ) -> Optional[RoutingDebugRecord]:
        """Attach optional trace['model_router'] to the newest record (post-native inference)."""
        if not model_router:
            return None
        with self._lock:
            if not self._deque:
                return None
            last = self._deque[-1]
            new_trace = dict(last.trace)
            new_trace["model_router"] = copy.deepcopy(model_router)
            updated = RoutingDebugRecord(
                timestamp=last.timestamp,
                session_id=last.session_id,
                turn_id=last.turn_id,
                query=last.query,
                route=last.route,
                route_pre_policy=last.route_pre_policy,
                strategy=last.strategy,
                trace_level=last.trace_level,
                top_intent=last.top_intent,
                top_score=last.top_score,
                summary=last.summary,
                trace=new_trace,
                decision=last.decision,
            )
            self._deque[-1] = updated
            return updated

    def merge_chat_contract_into_latest(
        self, chat_contract: Optional[dict[str, Any]]
    ) -> Optional[RoutingDebugRecord]:
        """Attach optional trace['chat_contract'] to the newest record (post-native inference)."""
        if not chat_contract:
            return None
        with self._lock:
            if not self._deque:
                return None
            last = self._deque[-1]
            new_trace = dict(last.trace)
            new_trace["chat_contract"] = copy.deepcopy(chat_contract)
            updated = RoutingDebugRecord(
                timestamp=last.timestamp,
                session_id=last.session_id,
                turn_id=last.turn_id,
                query=last.query,
                route=last.route,
                route_pre_policy=last.route_pre_policy,
                strategy=last.strategy,
                trace_level=last.trace_level,
                top_intent=last.top_intent,
                top_score=last.top_score,
                summary=last.summary,
                trace=new_trace,
                decision=last.decision,
            )
            self._deque[-1] = updated
            return updated

    def merge_engine_input_into_latest(
        self, engine_input: Optional[dict[str, Any]]
    ) -> Optional[RoutingDebugRecord]:
        """Attach optional trace['engine_input_trace'] to the newest record (post-native inference)."""
        if not engine_input:
            return None
        with self._lock:
            if not self._deque:
                return None
            last = self._deque[-1]
            new_trace = dict(last.trace)
            new_trace["engine_input_trace"] = copy.deepcopy(engine_input)
            updated = RoutingDebugRecord(
                timestamp=last.timestamp,
                session_id=last.session_id,
                turn_id=last.turn_id,
                query=last.query,
                route=last.route,
                route_pre_policy=last.route_pre_policy,
                strategy=last.strategy,
                trace_level=last.trace_level,
                top_intent=last.top_intent,
                top_score=last.top_score,
                summary=last.summary,
                trace=new_trace,
                decision=last.decision,
            )
            self._deque[-1] = updated
            return updated


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _infer_second_retrieval_intent(trace: dict[str, Any], top_intent: str) -> str:
    pool: dict[str, float] = {}
    for c in trace.get("losing_candidates") or []:
        ln = c.get("lane")
        if ln in ("memory", "rag"):
            pool[str(ln)] = max(pool.get(str(ln), 0.0), float(c.get("score") or 0.0))
    ws = trace.get("winning_signal") or {}
    if ws.get("lane") in ("memory", "rag"):
        ln = str(ws["lane"])
        pool[ln] = max(pool.get(ln, 0.0), float(ws.get("score") or 0.0))
    ranked = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)
    for ln, _ in ranked:
        if ln != top_intent:
            return ln
    return "runner-up"


def build_model_router_trace(native_engine: Any) -> Optional[dict[str, Any]]:
    """
    Read-only snapshot for routing debug UI (optional trace['model_router']).

    Returns None when native engine is unavailable or no router decision was recorded.
    """
    if native_engine is None or not hasattr(native_engine, "get_model_reasoning_telemetry"):
        return None
    try:
        tel = native_engine.get_model_reasoning_telemetry() or {}
    except Exception:
        return None
    selected = tel.get("router_selected_model")
    if selected is None or not str(selected).strip():
        return None
    selected_str = str(selected).strip()
    conf = _coerce_float(tel.get("router_confidence"))
    if conf is None:
        conf = 0.0
    conf = max(0.0, min(1.0, float(conf)))

    scores_raw = tel.get("router_scores") or {}
    scores: dict[str, float] = {}
    if isinstance(scores_raw, dict):
        for k, v in scores_raw.items():
            fv = _coerce_float(v)
            if fv is not None:
                scores[str(k)] = fv

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    alternatives: list[str] = []
    for name, _sc in ranked:
        if name == selected_str:
            continue
        alternatives.append(name)
        if len(alternatives) >= 5:
            break

    reasons: list[str] = []
    rr = tel.get("router_reasoning")
    if isinstance(rr, list):
        reasons.extend(str(x) for x in rr if str(x).strip())
    elif rr is not None:
        reasons.append(str(rr))
    task = tel.get("router_task")
    if task and not any(str(task) in x for x in reasons):
        reasons.insert(0, f"matched_task={task}")

    loaded_bn = str(tel.get("model_basename") or "").strip()
    model_name = str(tel.get("model_name") or "").strip()
    signals: dict[str, Any] = {
        "router_task": tel.get("router_task"),
        "router_scores": dict(scores),
        "model_basename": loaded_bn or None,
        "model_name": model_name or None,
        "selected_aligned_with_loaded": bool(loaded_bn and selected_str == loaded_bn),
    }

    performance: dict[str, Any] = {}
    try:
        from core.model_performance_store import ModelPerformanceStore

        store = ModelPerformanceStore()
        perf = store.get(selected_str)
        if perf is None and model_name:
            perf = store.get(model_name)
        if perf is None and loaded_bn:
            perf = store.get(loaded_bn)
        if perf is not None:
            performance = {
                "quality": round(float(perf.avg_response_quality), 4),
                "failure_rate": round(float(perf.structural_failure_rate), 4),
                "latency_ms": round(float(perf.avg_latency) * 1000.0, 2),
                "total_requests": int(perf.total_requests),
            }
    except Exception:
        pass

    return {
        "selected_model": selected_str,
        "alternatives": alternatives,
        "reasons": reasons,
        "signals": signals,
        "performance": performance,
        "confidence": round(float(conf), 4),
    }


def build_chat_contract_trace(native_engine: Any) -> Optional[dict[str, Any]]:
    """
    Read-only snapshot for routing debug UI (optional trace['chat_contract']).
    """
    if native_engine is None or not hasattr(native_engine, "get_model_reasoning_telemetry"):
        return None
    try:
        tel = native_engine.get_model_reasoning_telemetry() or {}
    except Exception:
        return None
    blob = tel.get("chat_contract")
    if not isinstance(blob, dict):
        return None
    has_lock = bool(str(blob.get("format") or "").strip() or str(blob.get("model") or "").strip())
    has_safety = isinstance(blob.get("template_safety"), dict)
    if not has_lock and not has_safety:
        return None
    return copy.deepcopy(blob)


def build_engine_input_trace(native_engine: Any) -> Optional[dict[str, Any]]:
    """
    Read-only snapshot for routing debug UI (optional trace['engine_input_trace']).

    ``native_engine`` is reserved for future correlation; the last trace is read from
    ``EngineInputTracer`` when ``QUBE_ENGINE_INPUT_TRACE`` is enabled.
    """
    _ = native_engine
    if not engine_input_trace_enabled():
        return None
    last = EngineInputTracer().get_last()
    if last is None:
        return None
    return copy.deepcopy(engine_input_trace_to_public_dict(last))


def synthesize_trace_stub(decision: dict[str, Any]) -> dict[str, Any]:
    """Uniform trace shape for worker override paths (no cognitive ``trace``)."""
    route = str(decision.get("route", "none"))
    strategy = str(decision.get("strategy", "override"))
    reason_map = {
        "explicit_remember": "explicit_remember",
        "explicit_file_search": "explicit_file_search",
        "narrative_recap": "narrative_recap",
        "fallback": "fallback",
    }
    winning_reason = reason_map.get(strategy, strategy)
    return {
        "selected_route": route,
        "winning_reason": winning_reason,
        "winning_signal": {
            "lane": route,
            "score": 1.0,
            "threshold": 0.0,
            "source": "override",
        },
        "losing_candidates": [],
        "confidence": {
            "top_intent": None,
            "top_intent_source": "override",
            "top_score": 0.0,
            "second_best_score": 0.0,
            "margin": 0.0,
            "floor": float(MIN_CONFIDENCE_FLOOR),
            "ambiguity_margin": float(AMBIGUITY_MARGIN),
            "floor_applied": False,
            "ambiguity_applied": False,
            "tier2_active": False,
        },
        "tier3": {
            "band_active": False,
            "high_confidence_ceiling": 0.0,
            "damping": 0.0,
            "lane_bias": {"memory": 0.0, "rag": 0.0, "web": 0.0},
        },
        "tier4": {
            "active": False,
            "cluster_id": None,
            "cluster_size": 0,
            "cluster_dominant_route": None,
            "cluster_dominant_frequency": None,
            "cluster_oscillating": False,
        },
        "tier5_6": {
            "tier5_active": False,
            "policy": "override",
            "policy_reason": None,
            "tier6_active": False,
            "conflicts": [],
            "interpretation": "override",
        },
        "context": {
            "override_path": True,
            "strategy": strategy,
        },
    }


def build_route_summary(decision: dict[str, Any]) -> str:
    """Single-line summary from existing decision/trace fields only."""
    try:
        route_u = str(decision.get("route", "none")).upper()
        strategy = str(decision.get("strategy", ""))
        trace = decision.get("trace")
        top_score_v = _coerce_float(decision.get("top_score"))
        margin_v = _coerce_float(decision.get("confidence_margin"))
        top_intent = decision.get("top_intent")
        top_intent_s = str(top_intent) if top_intent is not None else ""

        def _score_suffix() -> str:
            parts = []
            if top_score_v is not None:
                parts.append(f"{top_score_v:.2f}")
            if margin_v is not None:
                parts.append(f"margin {margin_v:.2f}")
            if not parts:
                return ""
            return f" ({', '.join(parts)})"

        if not trace:
            if strategy == "explicit_remember":
                return "Explicit remember intent — no routing"
            if strategy == "explicit_file_search":
                return "File-search intent forced RAG"
            if strategy == "narrative_recap":
                return "Narrative recap forced MEMORY"
            if strategy == "fallback":
                return "NONE — cognitive router disabled (fallback)"
            return f"{route_u} (strategy={strategy})"

        winning_reason = str(trace.get("winning_reason", ""))
        recall_sc = _coerce_float(decision.get("recall_score"))
        recall_thr = _coerce_float(decision.get("recall_threshold"))
        web_thr = _coerce_float(decision.get("internet_threshold"))
        complexity_sc = _coerce_float(decision.get("complexity_score"))

        if winning_reason == "internet_enabled":
            ts = top_score_v if top_score_v is not None else 0.0
            wt = web_thr if web_thr is not None else 0.0
            m = margin_v if margin_v is not None else 0.0
            return (
                f"WEB selected — embedding match (top {ts:.2f} > threshold {wt:.2f}, margin {m:.2f})"
            )
        if winning_reason == "single_memory":
            base = "MEMORY selected — strong signal"
            if top_score_v is not None and margin_v is not None:
                return f"{base} ({top_score_v:.2f}, margin {margin_v:.2f})"
            return base + _score_suffix()
        if winning_reason == "single_rag":
            base = "RAG selected — strong embedding signal"
            if top_score_v is not None and margin_v is not None:
                return f"{base} ({top_score_v:.2f}, margin {margin_v:.2f})"
            return base + _score_suffix()
        if winning_reason == "dual_threshold_hybrid":
            ts = top_score_v if top_score_v is not None else 0.0
            return f"HYBRID — both memory and rag cleared their thresholds (top {ts:.2f})"
        if winning_reason == "ambiguity_upgrade_to_hybrid":
            second_intent = _infer_second_retrieval_intent(trace, top_intent_s)
            m = margin_v if margin_v is not None else 0.0
            return (
                f"HYBRID — ambiguity between {top_intent_s} and {second_intent} "
                f"(margin {m:.2f} < {AMBIGUITY_MARGIN:.2f})"
            )
        if winning_reason == "recall_override_hybrid":
            rs = recall_sc if recall_sc is not None else 0.0
            rt = recall_thr if recall_thr is not None else 0.0
            return f"HYBRID — recall intent override (recall {rs:.2f} >= {rt:.2f})"
        if winning_reason == "complexity_forced_hybrid":
            if complexity_sc is not None:
                return f"HYBRID — high complexity ({complexity_sc:.2f} > 0.75)"
            return "HYBRID — high complexity (estimated score > 0.75)"
        if winning_reason == "confidence_floor_downgrade_to_none":
            ts = top_score_v if top_score_v is not None else 0.0
            return (
                f"NONE — confidence floor downgraded {top_intent_s} "
                f"(top {ts:.2f} < floor {MIN_CONFIDENCE_FLOOR:.2f})"
            )
        if winning_reason == "no_lane_cleared_threshold":
            ts = top_score_v if top_score_v is not None else 0.0
            return f"NONE — no lane cleared its threshold (best {ts:.2f})"
        if winning_reason == "hybrid_unknown":
            return f"HYBRID — unknown hybrid branch (top {top_intent_s}){_score_suffix()}"
        if winning_reason == "unknown_route":
            return f"{route_u} — unknown_route{_score_suffix()}"

        return f"{route_u} (strategy={strategy}, reason={winning_reason})"
    except Exception:
        try:
            r = str(decision.get("route", "none")).upper()
            return f"{r} (summary unavailable)"
        except Exception:
            return "ROUTING (summary unavailable)"


def build_record(
    *,
    query: str,
    decision: dict[str, Any],
    session_id: Optional[str],
    turn_id: Optional[int],
    effective_route: Optional[str] = None,
) -> RoutingDebugRecord:
    """Assemble one record; ``decision`` is deep-copied for storage."""
    decision_copy = copy.deepcopy(decision)
    raw_route = decision_copy.get("route", "none")
    route_pre = str(decision_copy.get("route_pre_policy", raw_route)).lower()
    eff = (effective_route or str(raw_route)).lower()
    strategy = str(decision_copy.get("strategy", "adaptive_v4"))

    if decision_copy.get("trace"):
        trace_level = "full"
        trace = decision_copy["trace"].copy()
    else:
        trace_level = "minimal"
        trace = synthesize_trace_stub(decision_copy)

    ti = decision_copy.get("top_intent")
    top_intent = str(ti) if ti is not None else None
    ts = _coerce_float(decision_copy.get("top_score"))

    summary = build_route_summary(decision_copy)

    return RoutingDebugRecord(
        timestamp=time.time(),
        session_id=session_id,
        turn_id=turn_id,
        query=query,
        route=eff,
        route_pre_policy=route_pre,
        strategy=strategy,
        trace_level=trace_level,
        top_intent=top_intent,
        top_score=ts,
        summary=summary,
        trace=trace,
        decision=decision_copy,
    )
