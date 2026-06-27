"""External knowledge platform — evidence bundles and retrieval pipeline."""

from core.knowledge.types import (
    EvidenceBundle,
    EvidenceBundleSummary,
    EvidenceObject,
    WebRetrievalOutcome,
)
from core.knowledge.web_retrieval import run_web_retrieval

__all__ = [
    "EvidenceBundle",
    "EvidenceBundleSummary",
    "EvidenceObject",
    "WebRetrievalOutcome",
    "run_web_retrieval",
]
