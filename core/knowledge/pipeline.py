"""Evidence pipeline orchestrator (Phase 0: general web / DDG)."""

from __future__ import annotations

from typing import Any

from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline
from core.knowledge.types import EvidenceBundle, RetrievalContext


class EvidencePipeline:
    """Run adapter collection, relevance gate, and bundle assembly."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        return run_general_web_evidence_pipeline(ctx)
