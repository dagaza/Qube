"""Retrieval replay — re-execute and compare knowledge turns."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.knowledge.registry import get_knowledge_service
from core.knowledge.retrieval_profiles import get_profile_spec, normalize_profile_id
from core.knowledge.retrieval_records import RetrievalContextFingerprint
from core.knowledge.types import RetrievalContext, WebRetrievalOutcome
from core.knowledge.web_retrieval import run_v2_web_retrieval

logger = logging.getLogger("Qube.Knowledge.Replay")


@dataclass(frozen=True)
class ReplayResult:
    mode: str
    original_bundle_id: str
    replay_bundle_id: str | None
    outcome: WebRetrievalOutcome | None
    warnings: tuple[str, ...]
    compare: dict[str, Any] | None = None


def _fingerprint_from_record(record: dict[str, Any]) -> RetrievalContextFingerprint:
    import json

    raw = record.get("context_fingerprint_json")
    if raw:
        try:
            return RetrievalContextFingerprint.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    import json as _json

    adapters_raw = record.get("adapter_filter_json") or "[]"
    hashes_raw = record.get("connector_hashes_json") or "[]"
    try:
        adapters = tuple(str(a) for a in _json.loads(adapters_raw))
    except _json.JSONDecodeError:
        adapters = ()
    try:
        hashes = tuple(str(h) for h in _json.loads(hashes_raw))
    except _json.JSONDecodeError:
        hashes = ()
    return RetrievalContextFingerprint(
        query_raw=str(record.get("query_raw") or ""),
        query_resolved=str(record.get("query_resolved") or ""),
        knowledge_service=str(record.get("knowledge_service") or ""),
        preset_id=str(record.get("preset_id") or "") or None,
        adapter_filter=adapters,
        retrieval_profile=str(record.get("retrieval_profile") or "balanced"),
        connector_config_hashes=hashes,
    )


def compare_traces(
    original: dict[str, Any] | None,
    replayed: dict[str, Any] | None,
) -> dict[str, Any]:
    orig_ids = set(original.get("evidence_ids_kept") or []) if original else set()
    replay_ids = set(replayed.get("evidence_ids_kept") or []) if replayed else set()
    return {
        "evidence_added": sorted(replay_ids - orig_ids),
        "evidence_removed": sorted(orig_ids - replay_ids),
        "evidence_unchanged": sorted(orig_ids & replay_ids),
        "coverage_before": original.get("coverage") if original else None,
        "coverage_after": replayed.get("coverage") if replayed else None,
        "latency_before_ms": original.get("latency_ms") if original else None,
        "latency_after_ms": replayed.get("latency_ms") if replayed else None,
        "confidence_before": original.get("confidence") if original else None,
        "confidence_after": replayed.get("confidence") if replayed else None,
    }


def replay_from_record(
    record: dict[str, Any],
    *,
    mode: str = "current",
    retrieval_profile: str | None = None,
    preset_id: str | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    db: Any | None = None,
) -> ReplayResult:
    """Replay a stored retrieval.

    Modes:
      - current: re-run with current connector config and optional overrides
      - original: best-effort replay using stored fingerprint (config may have drifted)
    """
    fingerprint = _fingerprint_from_record(record)
    warnings: list[str] = []
    if mode == "original":
        warnings.append(
            "Replay Original is best-effort: live APIs, credentials, and caches may differ."
        )

    profile_id = normalize_profile_id(retrieval_profile or fingerprint.retrieval_profile)
    profile = get_profile_spec(profile_id)
    service_id = fingerprint.knowledge_service
    adapter_filter = fingerprint.adapter_filter or None
    effective_preset = preset_id if preset_id is not None else fingerprint.preset_id

    if preset_id is not None and preset_id != fingerprint.preset_id:
        warnings.append(f"Preset override: {fingerprint.preset_id} → {preset_id}")
    if retrieval_profile and retrieval_profile != fingerprint.retrieval_profile:
        warnings.append(
            f"Profile override: {fingerprint.retrieval_profile} → {retrieval_profile}"
        )

    service = get_knowledge_service(service_id)
    budget = profile.materialize_budget(service.default_budget())

    outcome = run_v2_web_retrieval(
        query=fingerprint.query_raw,
        semantic_query=fingerprint.query_resolved,
        embed_fn=embed_fn,
        knowledge_service=service_id,
        adapter_filter=adapter_filter,
        budget=budget,
        preset_id=effective_preset,
        retrieval_profile=profile_id,
        db=db,
    )

    replay_trace = None
    if outcome.bundle is not None:
        from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace

        replay_trace = serialize_retrieval_trace(
            build_retrieval_trace(outcome.bundle, relevance_diag=outcome.relevance_diag),
            sources=outcome.bundle.sources,
        )

    original_trace = {
        "evidence_ids_kept": [],
        "coverage": record.get("coverage"),
        "latency_ms": record.get("latency_ms"),
        "confidence": record.get("confidence"),
    }
    if replay_trace:
        cmp = compare_traces(original_trace, replay_trace)
    else:
        cmp = None
        warnings.append("Replay produced no evidence bundle.")

    return ReplayResult(
        mode=mode,
        original_bundle_id=str(record.get("bundle_id") or ""),
        replay_bundle_id=outcome.bundle.bundle_id if outcome.bundle else None,
        outcome=outcome,
        warnings=tuple(warnings),
        compare=cmp,
    )
