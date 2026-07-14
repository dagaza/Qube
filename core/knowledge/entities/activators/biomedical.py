"""Biomedical entity activator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.knowledge.types import EvidenceObject

_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(ace[\s-]?inhibitors?|acei|arb|sglt2|statins?|gliflozin)\b", re.I),
    re.compile(r"\bheart[\s-]failure\b|\bhfref\b|\bhfpef\b|\bstemi\b|\bnstemi\b", re.I),
    re.compile(
        r"\b(paradigm[\s-]?hf|emperor|dapa[\s-]?hf|declare[\s-]?timi|solvd|consensus)\b",
        re.I,
    ),
    re.compile(
        r"\b(lisinopril|enalapril|empagliflozin|dapagliflozin|canagliflozin|semaglutide)\b",
        re.I,
    ),
    re.compile(r"\b(clinical[\s-]trial|randomi[sz]ed|placebo|cardiovascular)\b", re.I),
)

_BIOMEDICAL_ADAPTERS = frozenset({"pubmed"})


@dataclass(frozen=True)
class BiomedicalActivator:
    id: str = "biomedical"
    pack_id: str = "biomedical"
    priority: int = 10
    enables: tuple[str, ...] = (
        "biomedical_drugs",
        "biomedical_conditions",
        "biomedical_trials",
    )
    query_patterns: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: _QUERY_PATTERNS
    )

    def matches_query(self, query: str) -> bool:
        text = query or ""
        return any(pattern.search(text) for pattern in self.query_patterns)

    def matches_source(self, source: EvidenceObject) -> bool:
        adapter = (source.adapter or "").strip().lower()
        if adapter in _BIOMEDICAL_ADAPTERS:
            return True
        doc_type = (source.document_type or "").strip().lower()
        return doc_type in {"journal_abstract", "clinical_trial"} and adapter in _BIOMEDICAL_ADAPTERS


BIOMEDICAL_ACTIVATOR = BiomedicalActivator()
