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
STRATEGY_GENERIC_PARALLEL = "generic_parallel_ranked"


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
    "crossref": 0.87,
    "semantic_scholar": 0.88,
    "repec": 0.88,
    "dblp": 0.84,
    "biorxiv": 0.78,
    "pubchem": 0.90,
    "inspire_hep": 0.88,
    "nasa_ads": 0.90,
    "arxiv": 0.72,
    "europe_pmc": 0.90,
    "socarxiv": 0.76,
    "ssrn": 0.82,
    "psyarxiv": 0.76,
    "noaa": 0.92,
    "nasa_earthdata": 0.91,
    "acm_dl": 0.86,
    "psycinfo": 0.91,
    "clinicaltrials_gov": 0.94,
    "openfda": 0.96,
    "world_bank": 0.94,
    "eurostat": 0.94,
    "usgs": 0.95,
    "usda_fdc": 0.93,
    "nist": 0.94,
    "ietf_rfc": 0.96,
    "bls": 0.95,
    "us_census": 0.95,
    "ieee_xplore": 0.90,
    "oecd": 0.95,
    "nice": 0.97,
    "cdc": 0.96,
    "who": 0.96,
    "ipcc": 0.97,
    "fao": 0.95,
    "usda": 0.94,
    "copernicus_cds": 0.95,
    "openreview": 0.87,
    "acl_anthology": 0.89,
    "chembl": 0.91,
    "uniprot": 0.93,
    "pdb": 0.92,
    "chemrxiv": 0.78,
    "uspto_patentsview": 0.90,
    "epo_espacenet": 0.91,
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
    title = str(row.get("title") or "").strip() or f"Finance source {index}"
    snippet = str(row.get("snippet") or "").strip()
    url = str(row.get("url") or "").strip() or None
    form = str(row.get("form") or "").strip()
    adapter = str(row.get("_adapter") or "sec_edgar")
    retrieval_method = str(row.get("retrieval_method") or _finance_retrieval_method(adapter))
    relevance = float(row.get("_web_token_overlap") or 0.82)
    authority = _finance_authority(adapter)
    venue = str(row.get("venue") or _finance_venue(adapter))
    document_type = str(row.get("document_type") or _finance_document_type(adapter))

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=str(
            row.get("accession_number")
            or row.get("series_id")
            or row.get("company_number")
            or row.get("symbol")
            or url
            or title[:96]
        ),
        adapter=adapter,
        retrieval_method=retrieval_method,
        title=title,
        excerpt=snippet,
        full_text=row.get("full_text") if isinstance(row.get("full_text"), str) else None,
        url=url,
        document_type=document_type,
        publication_date=row.get("publication_date"),
        venue=venue,
        authors=(),
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=authority,
        reliability_score=0.9,
        freshness_score=freshness_score(row.get("publication_date")),
        retrieved_at=retrieved_at,
        fetch_status="abstract",
        raw_metadata={
            "form": form,
            "company": row.get("company"),
            "cik": row.get("cik"),
            "report_date": row.get("report_date"),
            "series_id": row.get("series_id"),
            "frequency": row.get("frequency"),
            "units": row.get("units"),
            "company_number": row.get("company_number"),
            "company_status": row.get("company_status"),
            "symbol": row.get("symbol"),
            "region": row.get("region"),
            "currency": row.get("currency"),
        },
    )


def _finance_retrieval_method(adapter: str) -> str:
    return {
        "fred": "fred_series_search",
        "companies_house": "companies_house_search",
        "alpha_vantage": "alpha_vantage_symbol_search",
        "bloomberg_api": "bloomberg_instrument_search",
    }.get(adapter, "sec_submissions")


def _finance_venue(adapter: str) -> str:
    return {
        "fred": "FRED",
        "companies_house": "Companies House",
        "alpha_vantage": "Alpha Vantage",
        "bloomberg_api": "Bloomberg",
        "world_bank": "World Bank Open Data",
        "eurostat": "Eurostat",
        "bls": "BLS",
        "oecd": "OECD",
    }.get(adapter, "SEC EDGAR")


def _finance_document_type(adapter: str) -> str:
    return {
        "fred": "macro_series",
        "companies_house": "uk_company_registry",
        "alpha_vantage": "market_symbol",
        "bloomberg_api": "market_symbol",
        "world_bank": "statistical_indicator",
        "eurostat": "statistical_release",
        "bls": "statistical_release",
        "oecd": "statistical_release",
    }.get(adapter, "sec_filing")


def _finance_authority(adapter: str) -> float:
    return {
        "sec_edgar": 0.95,
        "fred": 0.92,
        "companies_house": 0.93,
        "alpha_vantage": 0.85,
        "bloomberg_api": 0.94,
        "world_bank": 0.94,
        "eurostat": 0.94,
        "bls": 0.95,
        "oecd": 0.95,
    }.get(adapter, 0.9)


def _compute_finance_coverage(
    sources: tuple[EvidenceObject, ...],
) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No finance sources found."
    adapters = {s.adapter for s in sources}
    if adapters == {"fred"}:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} FRED macro series matched the query.",
        )
    if adapters == {"companies_house"}:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} UK company registry match(es) from Companies House.",
        )
    if adapters == {"alpha_vantage"}:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} market symbol match(es) from Alpha Vantage.",
        )
    forms = {
        str((s.raw_metadata or {}).get("form") or s.title.split("—")[0].strip())
        for s in sources
        if s.adapter == "sec_edgar"
    }
    forms.discard("")
    if len(sources) >= 2 and len(forms) >= 2:
        return (
            COVERAGE_EXCELLENT,
            f"{len(sources)} finance source(s) across {len(forms)} SEC form types.",
        )
    if len(sources) >= 1:
        if len(adapters) > 1:
            labels = ", ".join(sorted(_finance_venue(a) for a in adapters))
            return (
                COVERAGE_ADEQUATE,
                f"{len(sources)} finance source(s) from {labels}.",
            )
        if "fred" in adapters:
            return (
                COVERAGE_ADEQUATE,
                f"{len(sources)} FRED macro series retrieved.",
            )
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} SEC filing(s) retrieved from EDGAR.",
        )
    return COVERAGE_POOR, "Limited finance source coverage."


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
    title = str(row.get("title") or "").strip() or f"Legal source {index}"
    snippet = str(row.get("snippet") or "").strip()
    url = str(row.get("url") or "").strip() or None
    relevance = float(row.get("_web_token_overlap") or 0.82)
    adapter = str(row.get("_adapter") or "courtlistener")
    authority = float(row.get("authority_score") or _legal_authority(adapter))
    retrieval_method = str(row.get("retrieval_method") or _legal_retrieval_method(adapter))
    document_type = str(row.get("document_type") or _legal_document_type(adapter))
    venue = str(row.get("venue") or _legal_venue(adapter))

    return EvidenceObject(
        id=f"ek_{index}",
        source_id=str(
            row.get("cluster_id")
            or row.get("celex")
            or row.get("case_id")
            or url
            or title[:96]
        ),
        adapter=adapter,
        retrieval_method=retrieval_method,
        title=title,
        excerpt=snippet,
        full_text=None,
        url=url,
        document_type=document_type,
        publication_date=row.get("publication_date"),
        venue=venue,
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
            "celex": row.get("celex"),
            "database_id": row.get("database_id"),
            "case_id": row.get("case_id"),
            "jurisdiction": row.get("jurisdiction"),
        },
    )


def _legal_retrieval_method(adapter: str) -> str:
    return {
        "courtlistener": "courtlistener_search",
        "eur_lex": "eur_lex_search",
        "canlii": "canlii_search",
        "bailii": "bailii_search",
        "congress_gov": "congress_gov_bill_search",
        "govinfo": "govinfo_search",
        "legislation_uk": "legislation_uk_search",
    }.get(adapter, "courtlistener_search")


def _legal_venue(adapter: str) -> str:
    return {
        "courtlistener": "CourtListener",
        "eur_lex": "EUR-Lex",
        "canlii": "CanLII",
        "bailii": "BAILII",
        "congress_gov": "Congress.gov",
        "govinfo": "GovInfo",
        "legislation_uk": "legislation.gov.uk",
    }.get(adapter, "CourtListener")


def _legal_document_type(adapter: str) -> str:
    return {
        "eur_lex": "eu_legal_act",
        "congress_gov": "federal_bill",
        "govinfo": "federal_publication",
        "legislation_uk": "uk_legislation",
    }.get(adapter, "court_opinion")


def _legal_authority(adapter: str) -> float:
    return {
        "courtlistener": 0.82,
        "eur_lex": 0.94,
        "canlii": 0.84,
        "bailii": 0.82,
        "congress_gov": 0.96,
        "govinfo": 0.95,
        "legislation_uk": 0.96,
    }.get(adapter, 0.82)


def _compute_legal_coverage(
    sources: tuple[EvidenceObject, ...],
) -> tuple[str, str]:
    if not sources:
        return COVERAGE_NONE, "No case law opinions found."
    adapters = {s.adapter for s in sources}
    if adapters == {"eur_lex"}:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} EU legal act(s) matched from EUR-Lex.",
        )
    if adapters == {"canlii"}:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} Canadian case(s) matched from CanLII.",
        )
    if adapters == {"bailii"}:
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} UK/Irish case(s) matched from BAILII.",
        )
    courts = {
        str((s.raw_metadata or {}).get("court_id") or s.venue)
        for s in sources
    }
    if len(sources) >= 2 and len(adapters) > 1:
        labels = ", ".join(sorted(_legal_venue(a) for a in adapters))
        return (
            COVERAGE_ADEQUATE,
            f"{len(sources)} legal source(s) from {labels}.",
        )
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
    """Build a legal-knowledge bundle from jurisdiction adapter rows."""
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


def _generic_row_to_evidence(
    row: dict[str, Any],
    *,
    index: int,
    retrieved_at: float,
) -> EvidenceObject:
    title = str(row.get("title") or "").strip() or f"Source {index}"
    snippet = str(row.get("snippet") or "").strip()
    url = str(row.get("url") or "").strip() or None
    if url and not url.startswith(("http://", "https://", "file:")):
        url = None
    adapter = str(row.get("_adapter") or "generic")
    relevance = float(row.get("_generic_relevance") or row.get("_scientific_relevance") or 0.5)
    source_id = url or f"{adapter}:{title[:96]}"
    return EvidenceObject(
        id=f"ek_{index}",
        source_id=source_id,
        adapter=adapter,
        retrieval_method=str(row.get("retrieval_method") or "api"),
        title=title,
        excerpt=snippet,
        full_text=row.get("full_text"),
        url=url,
        document_type=str(row.get("document_type") or "generic"),
        relevance_score=max(0.0, min(1.0, relevance)),
        authority_score=authority_score_for_url(url) if url else 0.4,
        reliability_score=max(0.0, min(1.0, relevance * 0.8)),
        retrieved_at=retrieved_at,
        fetch_status="snippet_only",
        raw_metadata={
            "source_kind": row.get("_source_kind"),
            "connector_type": row.get("_connector_type"),
            "config_hash": row.get("_config_hash"),
        },
    )


def build_generic_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    kept_rows: list[dict[str, Any]],
    rejected_count: int,
    latency_ms: float,
    knowledge_service: str,
    retrieval_strategy: str = STRATEGY_GENERIC_PARALLEL,
    adapter_calls: tuple[str, ...] = (),
    stop_reason: str = "budget_exhausted",
    ranking_profile: str = "generic",
    preset_id: str | None = None,
) -> EvidenceBundle:
    retrieved_at = time.time()
    sources = tuple(
        _generic_row_to_evidence(row, index=i, retrieved_at=retrieved_at)
        for i, row in enumerate(kept_rows, start=1)
    )
    coverage, coverage_rationale = _compute_coverage(sources)
    confidence = (
        sum(s.relevance_score for s in sources) / len(sources) if sources else 0.0
    )
    warnings: list[str] = []
    if preset_id:
        warnings.append(f"preset:{preset_id}")
    if ranking_profile != "generic":
        warnings.append(f"ranking_profile:{ranking_profile}")
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
        retrieval_strategy=retrieval_strategy,
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
