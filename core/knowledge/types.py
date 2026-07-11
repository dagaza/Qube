"""Evidence and retrieval types for the external knowledge platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

COVERAGE_EXCELLENT = "excellent"
COVERAGE_ADEQUATE = "adequate"
COVERAGE_POOR = "poor"
COVERAGE_NONE = "none"

SERVICE_GENERAL_WEB = "general_web"
SERVICE_TRUSTED_KNOWLEDGE = "trusted_knowledge"
SERVICE_SCIENTIFIC_EVIDENCE = "scientific_evidence"
SERVICE_WIKIPEDIA = "wikipedia"
SERVICE_INTERNAL_CORPUS = "internal_corpus"
SERVICE_FINANCE_KNOWLEDGE = "finance_knowledge"
SERVICE_LEGAL_KNOWLEDGE = "legal_knowledge"
SERVICE_PRESET_KNOWLEDGE = "preset_knowledge"


@dataclass(frozen=True)
class EvidenceConflict:
    topic: str
    positions: tuple[tuple[str, str], ...]
    severity: str  # "minor" | "material"


@dataclass(frozen=True)
class EvidenceObject:
    id: str
    source_id: str
    adapter: str
    retrieval_method: str

    title: str
    excerpt: str
    full_text: str | None
    url: str | None

    document_type: str
    publication_date: str | None = None
    venue: str | None = None
    authors: tuple[str, ...] = ()
    doi: str | None = None
    peer_reviewed: bool | None = None
    preprint: bool | None = None
    open_access: bool | None = None
    retracted: bool | None = None

    relevance_score: float = 0.0
    authority_score: float = 0.0
    reliability_score: float = 0.0
    freshness_score: float | None = None

    retrieved_at: float = 0.0
    fetch_status: str = "snippet_only"
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    entity_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceBundleSummary:
    present: bool
    knowledge_service: str | None
    source_count: int
    confidence: float | None
    coverage: str | None
    has_conflicts: bool
    warnings: tuple[str, ...]
    source_types: tuple[str, ...]
    fetch_depth: str


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    query_raw: str
    query_resolved: str
    knowledge_service: str
    retrieval_strategy: str
    profile_version: str

    retrieved_at: float
    latency_ms: float

    confidence: float
    coverage: str
    coverage_rationale: str

    authority_summary: float
    reliability_summary: float
    diversity_summary: float

    sources: tuple[EvidenceObject, ...]
    rejected_count: int
    warnings: tuple[str, ...]
    conflicts: tuple[EvidenceConflict, ...]

    stop_reason: str
    adapter_calls: tuple[str, ...]

    def summary_for_skills(self) -> EvidenceBundleSummary:
        source_types = tuple(
            sorted({s.document_type for s in self.sources if s.document_type})
        )
        if not self.sources:
            fetch_depth = "snippet_only"
        elif all(s.fetch_status == "snippet_only" for s in self.sources):
            fetch_depth = "snippet_only"
        elif all(s.fetch_status != "snippet_only" for s in self.sources):
            fetch_depth = "abstract"
        else:
            fetch_depth = "mixed"

        return EvidenceBundleSummary(
            present=bool(self.sources),
            knowledge_service=self.knowledge_service,
            source_count=len(self.sources),
            confidence=self.confidence if self.sources else None,
            coverage=self.coverage if self.sources else COVERAGE_NONE,
            has_conflicts=bool(self.conflicts),
            warnings=self.warnings,
            source_types=source_types,
            fetch_depth=fetch_depth,
        )


@dataclass(frozen=True)
class RetrievalBudget:
    max_results: int = 3
    max_adapter_calls: int = 1
    max_fetch_bytes: int = 0
    max_latency_ms: int = 5000


@dataclass(frozen=True)
class RetrievalContext:
    query: str
    semantic_query: str
    knowledge_service: str = SERVICE_GENERAL_WEB
    query_vector: np.ndarray | None = None
    embed_fn: Callable[[str], np.ndarray] | None = None
    budget: RetrievalBudget = field(default_factory=RetrievalBudget)
    adapter_filter: tuple[str, ...] | None = None
    library_store: Any | None = None
    source_filter: str | None = None
    preset_id: str | None = None
    retrieval_profile: str = "balanced"


@dataclass(frozen=True)
class WebRetrievalOutcome:
    """Legacy-compatible web turn result plus optional v2 bundle."""

    web_results: list[dict[str, Any]] | None
    web_results_raw_for_audit: list[dict[str, Any]] | None
    web_results_kept_for_audit: list[dict[str, Any]] | None
    relevance_diag: dict[str, Any] | None
    skip_enrichment: bool
    bundle: EvidenceBundle | None
    latency_ms: float
