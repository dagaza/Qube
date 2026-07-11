"""RetrievalRecord — always-on minimal persistence for v2 knowledge turns."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import DatabaseManager
    from core.knowledge.types import EvidenceBundle, RetrievalContext

logger = logging.getLogger("Qube.Knowledge.Records")


@dataclass(frozen=True)
class RetrievalContextFingerprint:
    query_raw: str
    query_resolved: str
    knowledge_service: str
    preset_id: str | None
    adapter_filter: tuple[str, ...]
    retrieval_profile: str
    connector_config_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_raw": self.query_raw,
            "query_resolved": self.query_resolved,
            "knowledge_service": self.knowledge_service,
            "preset_id": self.preset_id,
            "adapter_filter": list(self.adapter_filter),
            "retrieval_profile": self.retrieval_profile,
            "connector_config_hashes": list(self.connector_config_hashes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RetrievalContextFingerprint:
        adapters = raw.get("adapter_filter") or []
        hashes = raw.get("connector_config_hashes") or []
        return cls(
            query_raw=str(raw.get("query_raw") or ""),
            query_resolved=str(raw.get("query_resolved") or ""),
            knowledge_service=str(raw.get("knowledge_service") or ""),
            preset_id=str(raw.get("preset_id") or "") or None,
            adapter_filter=tuple(str(a) for a in adapters if str(a).strip()),
            retrieval_profile=str(raw.get("retrieval_profile") or "balanced"),
            connector_config_hashes=tuple(str(h) for h in hashes if str(h).strip()),
        )


def build_context_fingerprint(
    ctx: RetrievalContext,
    *,
    retrieval_profile: str = "balanced",
) -> RetrievalContextFingerprint:
    from core.knowledge.configured_sources import load_configured_source

    hashes: list[str] = []
    for aid in ctx.adapter_filter or ():
        source = load_configured_source(str(aid))
        if source is not None:
            hashes.append(source.config_hash())

    return RetrievalContextFingerprint(
        query_raw=ctx.query,
        query_resolved=ctx.semantic_query or ctx.query,
        knowledge_service=ctx.knowledge_service,
        preset_id=ctx.preset_id,
        adapter_filter=tuple(ctx.adapter_filter or ()),
        retrieval_profile=retrieval_profile,
        connector_config_hashes=tuple(hashes),
    )


def save_retrieval_record(
    db: DatabaseManager | None,
    *,
    request_id: str,
    bundle: EvidenceBundle,
    context_fingerprint: RetrievalContextFingerprint,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> None:
    if db is None:
        return
    try:
        db.save_retrieval_record(
            request_id=request_id,
            bundle_id=bundle.bundle_id,
            session_id=session_id,
            turn_id=turn_id,
            query_raw=bundle.query_raw,
            query_resolved=bundle.query_resolved,
            knowledge_service=bundle.knowledge_service,
            retrieval_strategy=bundle.retrieval_strategy,
            preset_id=context_fingerprint.preset_id,
            adapter_filter_json=json.dumps(list(context_fingerprint.adapter_filter)),
            retrieval_profile=context_fingerprint.retrieval_profile,
            connector_hashes_json=json.dumps(list(context_fingerprint.connector_config_hashes)),
            context_fingerprint_json=json.dumps(context_fingerprint.to_dict()),
            evidence_count=len(bundle.sources),
            latency_ms=bundle.latency_ms,
            coverage=bundle.coverage,
            confidence=bundle.confidence,
        )
    except Exception as exc:
        logger.warning("Failed to save retrieval record: %s", exc)


def load_retrieval_record(db: DatabaseManager, *, bundle_id: str) -> dict[str, Any] | None:
    return db.get_retrieval_record(bundle_id=bundle_id)


def load_retrieval_record_by_request(
    db: DatabaseManager, *, request_id: str
) -> dict[str, Any] | None:
    return db.get_retrieval_record(request_id=request_id)
