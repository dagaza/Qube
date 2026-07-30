from rag.store import DocumentStore
import logging
import numpy as np

from core.chunking.rag_mmr import apply_mmr_to_rag_hits
from core.chunking.chunk_metadata import format_rag_source_header
from core.chunking.precision_rerank import apply_precision_rerank_to_rag_hits
from core.library_pro_features import precision_rerank_enabled
from core.retrieval_fusion import fuse_ranked_results
from core.memory_retrieval_policy import fts_query_token_overlap
from core.reindex_state import is_reindex_in_progress

logger = logging.getLogger("Qube.RAGTool")

MAX_CONTEXT_CHARS = 12000

# ============================================================
# T4.1: HARD semantic-relevance floor for RAG vector hits.
# ------------------------------------------------------------
# L2-normalized fastembed vectors — semantic_score (1 - distance) is a cosine proxy.
# Anything below this floor is topically unrelated — and yet top-k vector search will still return
# the nearest chunks in the library (by construction there is ALWAYS
# a "nearest" row, no matter how semantically distant it is).
#
# Without this gate, asking "Why is the sky blue?" against a library
# whose only document is "Project Omega (Blue Jay migration study)"
# returned the Omega chunk as SOURCE 1, because it was the least-far
# vector in the space. The LLM, given the citation-discipline system
# prompt, degenerated to a bare "[1]" token on that irrelevant source.
#
# This is the RAG-side mirror of mcp.memory_tool.MIN_SEMANTIC_SCORE
# (0.35). The RAG floor is slightly more permissive (0.30) because
# RAG chunks are longer than memory rows so their embeddings average
# over more text and tend to score lower on specific single-topic
# queries. Candidates dropped by this gate never reach llm_context,
# never appear in UI sources, and never enter fused ranking.
# ============================================================
MIN_RAG_SEMANTIC_SCORE = 0.30


def _escape_source_literal(source: str) -> str:
    return (source or "").replace("'", "''")


def _filter_results_by_source(results: list, source_filter: str) -> list:
    if not source_filter:
        return results
    return [
        doc
        for doc in results
        if (doc.get("source") or doc.get("filename") or "") == source_filter
    ]


def _filter_results_by_source_prefix(results: list, source_prefix_filter: str) -> list:
    if not source_prefix_filter:
        return results
    prefix = source_prefix_filter
    return [
        doc
        for doc in results
        if (doc.get("source") or doc.get("filename") or "").startswith(prefix)
    ]


def _escape_like_prefix(prefix: str) -> str:
    escaped = (prefix or "").replace("'", "''")
    return escaped.replace("%", "\\%").replace("_", "\\_")


def _source_scope_clause(
    *,
    source_filter: str | None,
    source_prefix_filter: str | None,
) -> str | None:
    if source_filter:
        esc = _escape_source_literal(source_filter)
        return f"source = '{esc}'"
    if source_prefix_filter:
        esc = _escape_like_prefix(source_prefix_filter)
        return f"source LIKE '{esc}%'"
    return None


def _exclude_help_corpus_results(results: list) -> list:
    from core.help_corpus_seed import is_help_corpus_source

    return [
        doc
        for doc in results
        if not is_help_corpus_source(doc.get("source") or doc.get("filename") or "")
    ]


def rag_search(
    query: str,
    query_vector: np.ndarray,
    store: DocumentStore,
    top_k: int = 5,
    *,
    source_filter: str | None = None,
    source_prefix_filter: str | None = None,
) -> dict:
    """
    RAG v2.3 — Contract-safe retrieval system.

    Design goals:
    - No assumptions about DB hybrid capabilities
    - No brittle NLP preprocessing
    - No UI contract drift
    - Safe fallback behavior across all configurations
    - Strict RAM + context enforcement
    """

    if source_filter and source_prefix_filter:
        raise ValueError("source_filter and source_prefix_filter are mutually exclusive")
    scope = ""
    if source_filter:
        scope = f" source={source_filter!r}"
    elif source_prefix_filter:
        scope = f" source_prefix={source_prefix_filter!r}"
    logger.info(f"[RAG v2.3] Query: {query}{scope}")

    if is_reindex_in_progress():
        logger.info("[RAG] Retrieval suppressed while library reprocessing is active.")
        return {"llm_context": "", "sources": []}

    try:
        # ============================================================
        # 1. RETRIEVAL CONTRACT LAYER (SAFE & EXPLICIT)
        # ============================================================

        vector_results = []
        text_results = []

        scope_clause = _source_scope_clause(
            source_filter=source_filter,
            source_prefix_filter=source_prefix_filter,
        )

        # --- VECTOR SEARCH (semantic channel) ---
        try:
            vector_query = store.table.search(query_vector)
            if scope_clause:
                vector_query = vector_query.where(scope_clause)
            vector_results = vector_query.limit(top_k * 2).to_list()
        except Exception as e:
            logger.error(f"[RAG] Vector search failed: {e}")

        # --- TEXT SEARCH (lexical fallback, optional capability) ---
        try:
            # NOTE:
            # Some setups require query_type="fts" to enable BM25.
            # If unsupported, this will safely fail and be ignored.
            text_query = store.table.search(query, query_type="fts")
            if scope_clause:
                text_query = text_query.where(scope_clause)
            text_results = text_query.limit(top_k * 2).to_list()
        except Exception as e:
            logger.debug(f"[RAG] FTS search unavailable: {e}")

        if source_filter:
            vector_results = _filter_results_by_source(vector_results, source_filter)
            text_results = _filter_results_by_source(text_results, source_filter)
        elif source_prefix_filter:
            vector_results = _filter_results_by_source_prefix(
                vector_results, source_prefix_filter
            )
            text_results = _filter_results_by_source_prefix(
                text_results, source_prefix_filter
            )
        else:
            vector_results = _exclude_help_corpus_results(vector_results)
            text_results = _exclude_help_corpus_results(text_results)

        had_vector_candidates = bool(vector_results)
        if not had_vector_candidates and text_results:
            filtered_fts: list = []
            for doc in text_results:
                body = (doc.get("text") or "").strip()
                if fts_query_token_overlap(query, body):
                    filtered_fts.append(doc)
                else:
                    logger.info(
                        "[RAG] dropped FTS hit below token overlap "
                        "(source=%s)",
                        doc.get("source") or doc.get("filename") or "?",
                    )
            text_results = filtered_fts

        # If everything fails, return safely
        if not vector_results and not text_results:
            if source_filter:
                try:
                    full_text = store.reconstruct_document(source_filter)
                except Exception as e:
                    logger.warning("[RAG] reconstruct_document failed for %s: %s", source_filter, e)
                    full_text = ""
                if full_text and full_text.strip():
                    snippet = full_text.strip()[:MAX_CONTEXT_CHARS]
                    logger.info(
                        "[RAG] Using full-document fallback for scoped source %s (%d chars)",
                        source_filter,
                        len(snippet),
                    )
                    return {
                        "llm_context": f"--- SOURCE 1: {source_filter} ---\n{snippet}",
                        "sources": [
                            {
                                "id": 1,
                                "filename": source_filter,
                                "content": snippet[:2000],
                                "type": "rag",
                                "chunk_id": f"{source_filter}::0",
                            }
                        ],
                    }
            logger.warning("[RAG] No retrieval results from any channel.")
            return {
                "llm_context": "",
                "sources": []
            }

        # ============================================================
        # 1.5 T4.1: HARD SEMANTIC-RELEVANCE GATE (vector channel)
        # ------------------------------------------------------------
        # Apply MIN_RAG_SEMANTIC_SCORE to each vector hit. If the
        # vector channel produced candidates but the gate drops ALL
        # of them, the FTS hits are also dropped: lexical matches
        # without semantic corroboration are almost always brittle
        # (e.g. FTS matching the bare word "blue" in a Blue Jay
        # migration study when the user asks about Rayleigh
        # scattering). If vector search was unavailable (empty by
        # exception path), we keep FTS as a fallback signal — the
        # gate only fires when vector results EXISTED.
        # ============================================================
        if had_vector_candidates:
            filtered_vector_results = []
            for doc in vector_results:
                distance = doc.get("_distance")
                if distance is None:
                    filtered_vector_results.append(doc)
                    continue
                try:
                    dist_val = float(distance)
                except (TypeError, ValueError):
                    filtered_vector_results.append(doc)
                    continue
                semantic_score = max(0.0, 1.0 - dist_val)
                if semantic_score < MIN_RAG_SEMANTIC_SCORE:
                    logger.info(
                        "[RAG] dropped chunk below relevance floor "
                        "(semantic=%.3f < %.2f; source=%s)",
                        semantic_score,
                        MIN_RAG_SEMANTIC_SCORE,
                        doc.get("source") or doc.get("filename") or "?",
                    )
                    continue
                filtered_vector_results.append(doc)

            if not filtered_vector_results:
                logger.info(
                    "[RAG] All %d vector candidates dropped by relevance floor "
                    "(floor=%.2f); suppressing FTS fallback to avoid brittle "
                    "lexical-only matches.",
                    len(vector_results),
                    MIN_RAG_SEMANTIC_SCORE,
                )
                return {
                    "llm_context": "",
                    "sources": []
                }
            vector_results = filtered_vector_results

        # ============================================================
        # 2. SAFE FUSION LAYER (DB-AGNOSTIC RANK MERGE)
        # ============================================================

        def _rag_doc_id(doc: dict) -> str:
            source = str(doc.get("source") or doc.get("filename") or "")
            raw_cid = doc.get("chunk_id")
            if raw_cid is None:
                raw_cid = doc.get("id")
            if source and raw_cid is not None:
                return f"{source}::{raw_cid}"
            return str(
                raw_cid
                or source
                or (doc.get("text") or "")[:64]
            )

        fused = fuse_ranked_results(
            vector_results,
            text_results,
            vector_weight=1.0,
            text_weight=0.8,
            doc_id_fn=_rag_doc_id,
        )
        ordered_results = [doc for doc, _channels in fused]
        ordered_results = apply_mmr_to_rag_hits(ordered_results, top_k=top_k)

        if precision_rerank_enabled():
            ordered_results = apply_precision_rerank_to_rag_hits(
                query_vector,
                ordered_results,
            )[:top_k]

        # ============================================================
        # 3. CONTEXT BUILDER (HARD SAFETY + UI CONTRACT ENFORCEMENT)
        # ============================================================

        context_blocks = []
        sources = []

        current_chars = 0

        for i, doc in enumerate(ordered_results[:top_k], start=1):

            text = (doc.get("text") or "").strip()
            source = doc.get("source") or doc.get("filename") or "Unknown Document"

            if not text:
                continue

            chunk_size = len(text)

            # HARD STOP: prevents memory / KV-cache overflow
            if current_chars + chunk_size > MAX_CONTEXT_CHARS:
                logger.warning(
                    f"[RAG] Context limit reached: "
                    f"{current_chars}/{MAX_CONTEXT_CHARS}"
                )
                break

            current_chars += chunk_size

            source_header = format_rag_source_header(source, doc.get("meta_json"))

            # LLM context block
            context_blocks.append(
                f"--- SOURCE {i}: {source_header} ---\n{text}"
            )

            # UI CONTRACT (NEVER CHANGE THIS SHAPE)
            # Phase B (memory enrichment): ``chunk_id`` is an ADDITIVE field —
            # ``id`` (1..n citation) and the other three contract fields are
            # unchanged. ``chunk_id`` is encoded as ``"<source>::<chunk_int>"``
            # so the memory tool can look the exact chunk back up later
            # (chunk_id alone is not unique across documents).
            raw_cid = doc.get("chunk_id")
            if raw_cid is None:
                raw_cid = doc.get("id")
            chunk_id_val = (
                f"{source}::{raw_cid}" if raw_cid is not None else None
            )
            sources.append({
                "id": i,
                "filename": source,
                "content": text,
                "type": "rag",
                "chunk_id": chunk_id_val,
            })

        # ============================================================
        # 4. FINAL RESPONSE
        # ============================================================

        logger.info(
            f"[RAG v2.3] Returned {len(context_blocks)} chunks | "
            f"chars={current_chars}"
        )

        return {
            "llm_context": "\n\n".join(context_blocks),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"[RAG v2.3] Fatal error: {e}")
        return {
            "llm_context": "",
            "sources": []
        }