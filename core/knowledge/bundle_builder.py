"""Assemble EvidenceBundle instances from adapter + gate outputs."""

from __future__ import annotations

import time
import uuid
from typing import Any

from core.knowledge.ranking.authority import authority_score_for_url, tier_label_for_url
from core.knowledge.ranking.freshness import freshness_score
from core.knowledge.types import (
    COVERAGE_ADEQUATE,
    COVERAGE_EXCELLENT,
    COVERAGE_NONE,
    COVERAGE_POOR,
    EvidenceBundle,
    EvidenceObject,
    SERVICE_GENERAL_WEB,
    SERVICE_INTERNAL_CORPUS,
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
    SERVICE_TRUSTED_KNOWLEDGE,
    SERVICE_WIKIPEDIA,
)

PROFILE_VERSION = "0.4.0"
STRATEGY_DDG_RELEVANCE_GATE = "ddg_serp_relevance_gate"
STRATEGY_WIKI_API_ALLOWLIST = "wiki_api_allowlist_ddg"
STRATEGY_SCIENTIFIC_PARALLEL = "pubmed_openalex_arxiv_ranked"
STRATEGY_INTERNAL_CORPUS = "lancedb_hybrid_library"
STRATEGY_FINANCE_SEC = "sec_edgar_submissions"
STRATEGY_LEGAL_COURTLISTENER = "courtlistener_search"


def _legacy_row_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or "").strip() or f"Web result {index}"
    snippet = str(row.get("snippet") or "").strip()
    url = str(row.get("url") or "").strip() or None
    if url and not url.startswith(("http://", "https://")):
        url = None

    token_overlap = float(row.get("_web_token_overlap") or 0.0)
    semantic = row.get("_web_semantic_score")
    relevance = float(semantic) if semantic is not None else token_overlap

    source_id = url or f"ddg:{title[:96]}"
    return EvidenceObject(
        id=f"ek_{index}",
        source_id=source_id,
        adapter="duckduckgo",
        retrieval_method="serp",
        title=title,
        excerpt=snippet,
        full_text=None,
        url=url,
        document_type="web_snippet",
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=0.35,
        reliability_score=max(0.0, min(1.0, relevance * 0.8)),
        retrieved_at=retrieved_at,
        fetch_status="snippet_only",
        raw_metadata={
            "token_overlap": row.get("_web_token_overlap"),
            "semantic_score": row.get("_web_semantic_score"),
        },
    )


def _compute_coverage(sources: tuple[EvidenceObject, ...]) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No sources passed the relevance gate."

    count = len(sources)
    avg_rel = sum(s.relevance_score for s in sources) / count
    if count >= 3 and avg_rel >= 0.25:
        return (
            COVERAGE_EXCELLENT,
            f"{count} SERP snippets retained with average relevance {avg_rel:.2f}.",
        )
    if count >= 2:
        return (
            COVERAGE_ADEQUATE,
            f"{count} SERP snippets retained; limited source diversity (SERP-only).",
        )
    return (
        COVERAGE_POOR,
        "Only one SERP snippet retained; coverage may be insufficient.",
    )


def _compute_confidence(sources: tuple[EvidenceObject, ...]) -> float:
    if not sources:
        return 0.0
    avg_rel = sum(s.relevance_score for s in sources) / len(sources)
    count_factor = min(1.0, len(sources) / 3.0)
    # Phase 0: snippet-only SERP — cap confidence modestly.
    return max(0.0, min(0.75, avg_rel * 0.6 + count_factor * 0.25))


def build_general_web_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    adapter_calls: tuple[str, ...] = ("duckduckgo",),
    stop_reason: str = "budget_exhausted",
) -> EvidenceBundle:
    """Build a Phase-0 general-web bundle from gated DDG rows."""
    retrieved_at = time.time()
    sources = tuple(
        _legacy_row_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_coverage(sources)
    confidence = _compute_confidence(sources)

    warnings: list[str] = []
    if sources and all(s.fetch_status == "snippet_only" for s in sources):
        warnings.append("serp_snippet_only")

    authority_summary = (
        sum(s.authority_score for s in sources) / len(sources) if sources else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in sources) / len(sources) if sources else 0.0
    )
    diversity_summary = min(1.0, len({s.adapter for s in sources}) / 3.0)

    return EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query_raw,
        query_resolved=query_resolved,
        knowledge_service=SERVICE_GENERAL_WEB,
        retrieval_strategy=STRATEGY_DDG_RELEVANCE_GATE,
        profile_version=PROFILE_VERSION,
        retrieved_at=retrieved_at,
        latency_ms=latency_ms,
        confidence=confidence,
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=sources,
        rejected_count=rejected_count,
        warnings=tuple(warnings),
        conflicts=(),
        stop_reason=stop_reason if sources else "no_evidence",
        adapter_calls=adapter_calls,
    )


def build_empty_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    latency_ms: float,
    rejected_count: int = 0,
    stop_reason: str = "no_evidence",
    knowledge_service: str = SERVICE_GENERAL_WEB,
) -> EvidenceBundle:
    """Empty bundle after sentinel drop, gate failure, or no results."""
    if knowledge_service == SERVICE_TRUSTED_KNOWLEDGE:
        return build_trusted_knowledge_bundle(
            query_raw=query_raw,
            query_resolved=query_resolved,
            kept_rows=[],
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
    if knowledge_service == SERVICE_WIKIPEDIA:
        return build_trusted_knowledge_bundle(
            query_raw=query_raw,
            query_resolved=query_resolved,
            kept_rows=[],
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            knowledge_service=SERVICE_WIKIPEDIA,
        )
    if knowledge_service == SERVICE_SCIENTIFIC_EVIDENCE:
        return build_scientific_evidence_bundle(
            query_raw=query_raw,
            query_resolved=query_resolved,
            kept_rows=[],
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
    if knowledge_service == SERVICE_INTERNAL_CORPUS:
        return build_internal_corpus_bundle(
            query_raw=query_raw,
            query_resolved=query_resolved,
            kept_rows=[],
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
    if knowledge_service == SERVICE_FINANCE_KNOWLEDGE:
        return build_finance_knowledge_bundle(
            query_raw=query_raw,
            query_resolved=query_resolved,
            kept_rows=[],
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
    if knowledge_service == SERVICE_LEGAL_KNOWLEDGE:
        return build_legal_knowledge_bundle(
            query_raw=query_raw,
            query_resolved=query_resolved,
            kept_rows=[],
            rejected_count=rejected_count,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )
    return build_general_web_bundle(
        query_raw=query_raw,
        query_resolved=query_resolved,
        kept_rows=[],
        rejected_count=rejected_count,
        latency_ms=latency_ms,
        stop_reason=stop_reason,
    )


def _trusted_row_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or "").strip() or f"Source {index}"
    snippet = str(row.get("snippet") or "").strip()
    full_text = row.get("full_text")
    if isinstance(full_text, str):
        full_text = full_text.strip() or None
    else:
        full_text = None
    url = str(row.get("url") or "").strip() or None
    if url and not url.startswith(("http://", "https://")):
        url = None

    is_wiki = bool(row.get("_wiki_source"))
    adapter = "wikipedia_api" if is_wiki else "duckduckgo"
    retrieval_method = "api_extract" if is_wiki else "serp"
    document_type = "encyclopedia" if is_wiki else tier_label_for_url(url)

    token_overlap = float(row.get("_web_token_overlap") or 0.0)
    semantic = row.get("_web_semantic_score")
    if is_wiki:
        relevance = 0.85
    else:
        relevance = float(semantic) if semantic is not None else token_overlap

    authority = authority_score_for_url(url)
    fetch_status = "abstract" if full_text else "snippet_only"
    source_id = url or f"{adapter}:{title[:96]}"
    reliability = max(0.0, min(1.0, relevance * 0.85))
    fresh = freshness_score(row.get("publication_date"))

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=source_id,
        adapter=adapter,
        retrieval_method=retrieval_method,
        title=title,
        excerpt=snippet,
        full_text=full_text,
        url=url,
        document_type=document_type,
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=authority,
        reliability_score=reliability,
        freshness_score=fresh,
        retrieved_at=retrieved_at,
        fetch_status=fetch_status,
        raw_metadata={
            "token_overlap": row.get("_web_token_overlap"),
            "semantic_score": row.get("_web_semantic_score"),
            "wiki_pageid": row.get("pageid"),
        },
    )


def _compute_trusted_coverage(sources: tuple[EvidenceObject, ...]) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No trusted sources found."

    count = len(sources)
    avg_auth = sum(s.authority_score for s in sources) / count
    has_wiki = any(s.adapter == "wikipedia_api" for s in sources)
    if count >= 2 and avg_auth >= 0.8:
        return (
            COVERAGE_EXCELLENT,
            f"{count} trusted sources with average authority {avg_auth:.2f}.",
        )
    if has_wiki or count >= 2:
        return (
            COVERAGE_ADEQUATE,
            f"{count} trusted source(s); {'Wikipedia present' if has_wiki else 'allowlisted web only'}.",
        )
    return (
        COVERAGE_POOR,
        "Single trusted source; corroboration may be limited.",
    )


def _compute_trusted_confidence(sources: tuple[EvidenceObject, ...]) -> float:
    if not sources:
        return 0.0
    avg_auth = sum(s.authority_score for s in sources) / len(sources)
    has_abstract = any(s.fetch_status != "snippet_only" for s in sources)
    depth_bonus = 0.12 if has_abstract else 0.0
    count_factor = min(1.0, len(sources) / 2.0)
    return max(0.0, min(0.92, avg_auth * 0.55 + count_factor * 0.25 + depth_bonus))


def build_trusted_knowledge_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    adapter_calls: tuple[str, ...] = ("wikipedia_api",),
    stop_reason: str = "budget_exhausted",
    knowledge_service: str = SERVICE_TRUSTED_KNOWLEDGE,
) -> EvidenceBundle:
    """Build a Phase-1 trusted-knowledge bundle from wiki + allowlisted rows."""
    retrieved_at = time.time()
    sources = tuple(
        _trusted_row_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_trusted_coverage(sources)
    confidence = _compute_trusted_confidence(sources)

    warnings: list[str] = []
    if sources and not any(s.adapter == "wikipedia_api" for s in sources):
        warnings.append("no_wikipedia_hit")
    if sources and all(s.fetch_status == "snippet_only" for s in sources):
        warnings.append("snippet_only")

    authority_summary = (
        sum(s.authority_score for s in sources) / len(sources) if sources else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in sources) / len(sources) if sources else 0.0
    )
    diversity_summary = min(1.0, len({s.adapter for s in sources}) / 2.0)

    return EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query_raw,
        query_resolved=query_resolved,
        knowledge_service=knowledge_service,
        retrieval_strategy=STRATEGY_WIKI_API_ALLOWLIST,
        profile_version=PROFILE_VERSION,
        retrieved_at=retrieved_at,
        latency_ms=latency_ms,
        confidence=confidence,
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=sources,
        rejected_count=rejected_count,
        warnings=tuple(warnings),
        conflicts=(),
        stop_reason=stop_reason if sources else "no_evidence",
        adapter_calls=adapter_calls,
    )


_SCIENTIFIC_AUTHORITY = {
    "pubmed": 0.92,
    "openalex": 0.86,
    "repec": 0.88,
    "dblp": 0.84,
    "arxiv": 0.72,
}


def _scientific_row_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or "").strip() or f"Source {index}"
    snippet = str(row.get("snippet") or "").strip()
    full_text = row.get("full_text")
    if isinstance(full_text, str):
        full_text = full_text.strip() or None
    else:
        full_text = None
    url = str(row.get("url") or "").strip() or None
    adapter = str(row.get("_adapter") or "pubmed")
    authors_raw = row.get("authors") or ()
    authors = tuple(str(a).strip() for a in authors_raw if str(a).strip())
    relevance = float(row.get("_scientific_relevance") or 0.75)
    authority = float(_SCIENTIFIC_AUTHORITY.get(adapter, 0.7))
    if row.get("peer_reviewed"):
        authority = max(authority, 0.85)
    if row.get("preprint"):
        authority = min(authority, 0.75)

    reliability = float(row.get("_reliability_score") or max(0.0, min(1.0, authority * 0.85)))
    fresh = freshness_score(row.get("publication_date"))

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=str(row.get("doi") or url or f"{adapter}:{title[:96]}"),
        adapter=adapter,
        retrieval_method="abstract",
        title=title,
        excerpt=snippet,
        full_text=full_text,
        url=url,
        document_type=str(row.get("document_type") or "journal_abstract"),
        publication_date=row.get("publication_date"),
        venue=row.get("venue"),
        authors=authors,
        doi=row.get("doi"),
        peer_reviewed=row.get("peer_reviewed"),
        preprint=row.get("preprint"),
        open_access=row.get("open_access"),
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=authority,
        reliability_score=max(0.0, min(1.0, authority * 0.9)),
        retrieved_at=retrieved_at,
        fetch_status="abstract" if full_text else "snippet_only",
        raw_metadata={
            "pmid": row.get("pmid"),
            "arxiv_id": row.get("arxiv_id"),
        },
    )


def _compute_scientific_coverage(
    sources: tuple[EvidenceObject, ...],
) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No scientific sources found."
    abstract_count = sum(1 for s in sources if s.fetch_status == "abstract")
    adapters = {s.adapter for s in sources}
    if len(sources) >= 3 and abstract_count >= 2 and len(adapters) >= 2:
        return (
            COVERAGE_EXCELLENT,
            f"{len(sources)} sources across {len(adapters)} scientific indexes; "
            f"{abstract_count} with abstracts.",
        )
    if abstract_count >= 1 and len(sources) >= 2:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} scientific source(s); {abstract_count} abstract(s) retrieved.",
        )
    if abstract_count >= 1:
        return (
            COVERAGE_POOR,
            "Single abstract retrieved; corroboration may be limited.",
        )
    return (
        COVERAGE_POOR,
        "Scientific hits lacked abstracts; snippet-only coverage.",
    )


def _compute_scientific_confidence(sources: tuple[EvidenceObject, ...]) -> float:
    if not sources:
        return 0.0
    avg_auth = sum(s.authority_score for s in sources) / len(sources)
    abstract_ratio = sum(1 for s in sources if s.fetch_status == "abstract") / len(
        sources
    )
    adapter_diversity = min(1.0, len({s.adapter for s in sources}) / 3.0)
    return max(
        0.0,
        min(
            0.94,
            avg_auth * 0.5 + abstract_ratio * 0.25 + adapter_diversity * 0.2,
        ),
    )


def build_scientific_evidence_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    adapter_calls: tuple[str, ...] = ("pubmed", "openalex", "arxiv"),
    stop_reason: str = "budget_exhausted",
    medical_query: bool = False,
    knowledge_service: str = SERVICE_SCIENTIFIC_EVIDENCE,
) -> EvidenceBundle:
    """Build a Phase-2 scientific-evidence bundle from adapter rows."""
    retrieved_at = time.time()
    sources = tuple(
        _scientific_row_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_scientific_coverage(sources)
    confidence = _compute_scientific_confidence(sources)

    warnings: list[str] = []
    snippet_only = [s for s in sources if s.fetch_status == "snippet_only"]
    if snippet_only:
        warnings.append(f"abstract_only_missing_for_{len(snippet_only)}_sources")
    if sources and len({s.adapter for s in sources}) == 1:
        warnings.append("single_index_only")
    if any(s.preprint for s in sources):
        warnings.append("preprint_included")
    if medical_query:
        warnings.append("medical_disclaimer")

    authority_summary = (
        sum(s.authority_score for s in sources) / len(sources) if sources else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in sources) / len(sources) if sources else 0.0
    )
    diversity_summary = min(1.0, len({s.adapter for s in sources}) / 3.0)

    return EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query_raw,
        query_resolved=query_resolved,
        knowledge_service=knowledge_service,
        retrieval_strategy=STRATEGY_SCIENTIFIC_PARALLEL,
        profile_version=PROFILE_VERSION,
        retrieved_at=retrieved_at,
        latency_ms=latency_ms,
        confidence=confidence,
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=sources,
        rejected_count=rejected_count,
        warnings=tuple(warnings),
        conflicts=(),
        stop_reason=stop_reason if sources else "no_evidence",
        adapter_calls=adapter_calls,
    )


def _library_chunk_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or row.get("source") or "").strip() or f"Library chunk {index}"
    snippet = str(row.get("snippet") or "").strip()
    full_text = row.get("full_text")
    if isinstance(full_text, str):
        full_text = full_text.strip() or None
    else:
        full_text = None

    semantic = row.get("_library_semantic_score")
    relevance = float(semantic) if semantic is not None else 0.75
    chunk_id = str(row.get("chunk_id") or "").strip()
    source_name = str(row.get("source") or title).strip()
    source_id = chunk_id or f"library:{source_name[:96]}"

    content_len = len(full_text or snippet)
    fetch_status = "full_text" if content_len > 400 else "snippet"

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=source_id,
        adapter="lancedb_library",
        retrieval_method="hybrid_vector_fts",
        title=title,
        excerpt=snippet[:2000],
        full_text=full_text,
        url=None,
        document_type="library_chunk",
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=0.92,
        reliability_score=max(0.0, min(1.0, relevance * 0.9)),
        freshness_score=0.85,
        retrieved_at=retrieved_at,
        fetch_status=fetch_status,
        raw_metadata={
            "chunk_id": chunk_id or None,
            "source": source_name,
            "semantic_score": semantic,
        },
    )


def _compute_internal_corpus_coverage(
    sources: tuple[EvidenceObject, ...],
) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No library chunks matched the query."

    count = len(sources)
    avg_rel = sum(s.relevance_score for s in sources) / count
    if count >= 3 and avg_rel >= 0.35:
        return (
            COVERAGE_EXCELLENT,
            f"{count} library chunks with average relevance {avg_rel:.2f}.",
        )
    if count >= 2:
        return (
            COVERAGE_ADEQUATE,
            f"{count} library chunk(s) retrieved from your indexed documents.",
        )
    return (
        COVERAGE_POOR,
        "Single library chunk; corroboration may be limited.",
    )


def _compute_internal_corpus_confidence(sources: tuple[EvidenceObject, ...]) -> float:
    if not sources:
        return 0.0
    avg_rel = sum(s.relevance_score for s in sources) / len(sources)
    count_factor = min(1.0, len(sources) / 3.0)
    return max(0.0, min(0.95, avg_rel * 0.5 + count_factor * 0.35 + 0.1))


def build_internal_corpus_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    adapter_calls: tuple[str, ...] = ("lancedb_library",),
    stop_reason: str = "budget_exhausted",
    knowledge_service: str = SERVICE_INTERNAL_CORPUS,
) -> EvidenceBundle:
    """Build an internal-corpus bundle from LanceDB library chunk rows."""
    retrieved_at = time.time()
    sources = tuple(
        _library_chunk_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_internal_corpus_coverage(sources)
    confidence = _compute_internal_corpus_confidence(sources)

    warnings: list[str] = []
    if sources and all(s.fetch_status == "snippet" for s in sources):
        warnings.append("snippet_only")

    authority_summary = (
        sum(s.authority_score for s in sources) / len(sources) if sources else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in sources) / len(sources) if sources else 0.0
    )
    diversity_summary = min(1.0, len({s.source_id.split(":")[0] for s in sources}) / 3.0)

    return EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query_raw,
        query_resolved=query_resolved,
        knowledge_service=knowledge_service,
        retrieval_strategy=STRATEGY_INTERNAL_CORPUS,
        profile_version=PROFILE_VERSION,
        retrieved_at=retrieved_at,
        latency_ms=latency_ms,
        confidence=confidence,
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=sources,
        rejected_count=rejected_count,
        warnings=tuple(warnings),
        conflicts=(),
        stop_reason=stop_reason,
        adapter_calls=adapter_calls,
    )


def _finance_row_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or "").strip() or f"SEC filing {index}"
    snippet = str(row.get("snippet") or "").strip()
    url = str(row.get("url") or "").strip() or None
    form = str(row.get("form") or "").strip()
    relevance = float(row.get("_web_token_overlap") or 0.82)

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=str(row.get("accession_number") or url or title[:96]),
        adapter=str(row.get("_adapter") or "sec_edgar"),
        retrieval_method="sec_submissions",
        title=title,
        excerpt=snippet,
        full_text=None,
        url=url,
        document_type=str(row.get("document_type") or "sec_filing"),
        publication_date=row.get("publication_date"),
        venue=str(row.get("venue") or "SEC EDGAR"),
        authors=(),
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=0.95,
        reliability_score=0.9,
        freshness_score=freshness_score(row.get("publication_date")),
        retrieved_at=retrieved_at,
        fetch_status="abstract",
        raw_metadata={
            "form": form,
            "company": row.get("company"),
            "cik": row.get("cik"),
            "report_date": row.get("report_date"),
        },
    )


def _compute_finance_coverage(
    sources: tuple[EvidenceObject, ...],
) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No SEC filings found."
    forms = {str((s.raw_metadata or {}).get("form") or s.title.split("—")[0].strip()) for s in sources}
    if len(sources) >= 2 and len(forms) >= 2:
        return (
            COVERAGE_EXCELLENT,
            f"{len(sources)} SEC filing(s) across {len(forms)} form types.",
        )
    if len(sources) >= 1:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} SEC filing(s) retrieved from EDGAR.",
        )
    return COVERAGE_POOR, "Limited SEC filing coverage."


def _compute_finance_confidence(sources: tuple[EvidenceObject, ...]) -> float:
    if not sources:
        return 0.0
    avg_auth = sum(s.authority_score for s in sources) / len(sources)
    count_factor = min(1.0, len(sources) / 2.0)
    return max(0.0, min(0.9, avg_auth * 0.6 + count_factor * 0.3))


def build_finance_knowledge_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    adapter_calls: tuple[str, ...] = ("sec_edgar",),
    stop_reason: str = "budget_exhausted",
    knowledge_service: str = SERVICE_FINANCE_KNOWLEDGE,
) -> EvidenceBundle:
    """Build a finance-knowledge bundle from SEC EDGAR rows."""
    retrieved_at = time.time()
    sources = tuple(
        _finance_row_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_finance_coverage(sources)
    confidence = _compute_finance_confidence(sources)

    warnings: list[str] = ["not_financial_advice"]
    if not sources:
        warnings.append("no_sec_filings")

    authority_summary = (
        sum(s.authority_score for s in sources) / len(sources) if sources else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in sources) / len(sources) if sources else 0.0
    )
    diversity_summary = min(1.0, len({s.adapter for s in sources}))

    return EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query_raw,
        query_resolved=query_resolved,
        knowledge_service=knowledge_service,
        retrieval_strategy=STRATEGY_FINANCE_SEC,
        profile_version=PROFILE_VERSION,
        retrieved_at=retrieved_at,
        latency_ms=latency_ms,
        confidence=confidence,
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=sources,
        rejected_count=rejected_count,
        warnings=tuple(dict.fromkeys(warnings)),
        conflicts=(),
        stop_reason=stop_reason if sources else "no_evidence",
        adapter_calls=adapter_calls,
    )


def _legal_row_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or "").strip() or f"Court opinion {index}"
    snippet = str(row.get("snippet") or "").strip()
    url = str(row.get("url") or "").strip() or None
    relevance = float(row.get("_web_token_overlap") or 0.82)
    authority = float(row.get("authority_score") or 0.82)

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=str(row.get("cluster_id") or url or title[:96]),
        adapter=str(row.get("_adapter") or "courtlistener"),
        retrieval_method="courtlistener_search",
        title=title,
        excerpt=snippet,
        full_text=None,
        url=url,
        document_type=str(row.get("document_type") or "court_opinion"),
        publication_date=row.get("publication_date"),
        venue=str(row.get("venue") or "CourtListener"),
        authors=(),
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=max(0.0, min(1.0, authority)),
        reliability_score=0.88,
        freshness_score=freshness_score(row.get("publication_date")),
        retrieved_at=retrieved_at,
        fetch_status="abstract",
        raw_metadata={
            "court": row.get("court"),
            "court_id": row.get("court_id"),
            "citation": row.get("citation"),
            "docket_number": row.get("docket_number"),
            "judge": row.get("judge"),
        },
    )


def _compute_legal_coverage(
    sources: tuple[EvidenceObject, ...],
) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No case law opinions found."
    courts = {
        str((s.raw_metadata or {}).get("court_id") or s.venue)
        for s in sources
    }
    if len(sources) >= 2 and len(courts) >= 2:
        return COVERAGE_EXCELLENT, "Multiple opinions from distinct courts."
    if len(sources) >= 2:
        return COVERAGE_ADEQUATE, "Multiple related opinions retrieved."
    return COVERAGE_ADEQUATE, "Single opinion retrieved."


def _compute_legal_confidence(sources: tuple[EvidenceObject, ...]) -> float:
    if not sources:
        return 0.0
    avg_auth = sum(s.authority_score for s in sources) / len(sources)
    count_factor = min(1.0, len(sources) / 2.0)
    return max(0.0, min(0.9, avg_auth * 0.6 + count_factor * 0.3))


def build_legal_knowledge_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    adapter_calls: tuple[str, ...] = ("courtlistener",),
    stop_reason: str = "budget_exhausted",
    knowledge_service: str = SERVICE_LEGAL_KNOWLEDGE,
) -> EvidenceBundle:
    """Build a legal-knowledge bundle from CourtListener rows."""
    retrieved_at = time.time()
    sources = tuple(
        _legal_row_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_legal_coverage(sources)
    confidence = _compute_legal_confidence(sources)

    warnings: list[str] = ["not_legal_advice"]
    if not sources:
        warnings.append("no_case_law")

    authority_summary = (
        sum(s.authority_score for s in sources) / len(sources) if sources else 0.0
    )
    reliability_summary = (
        sum(s.reliability_score for s in sources) / len(sources) if sources else 0.0
    )
    diversity_summary = min(1.0, len({s.adapter for s in sources}))

    return EvidenceBundle(
        bundle_id=str(uuid.uuid4()),
        query_raw=query_raw,
        query_resolved=query_resolved,
        knowledge_service=knowledge_service,
        retrieval_strategy=STRATEGY_LEGAL_COURTLISTENER,
        profile_version=PROFILE_VERSION,
        retrieved_at=retrieved_at,
        latency_ms=latency_ms,
        confidence=confidence,
        coverage=coverage,
        coverage_rationale=coverage_rationale,
        authority_summary=authority_summary,
        reliability_summary=reliability_summary,
        diversity_summary=diversity_summary,
        sources=sources,
        rejected_count=rejected_count,
        warnings=tuple(dict.fromkeys(warnings)),
        conflicts=(),
        stop_reason=stop_reason if sources else "no_evidence",
        adapter_calls=adapter_calls,
    )
