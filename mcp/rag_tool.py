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


def _exclude_memory_results(results: list) -> list:
    """Drop memory rows: memories live in the same table but are searched by ``memory_search``."""
    from core.memory_source_migration import MEMORY_SOURCE_PREFIX

    return [
        doc
        for doc in results
        if not str(doc.get("source") or doc.get("filename") or "").startswith(MEMORY_SOURCE_PREFIX)
    ]


def _shadow_row_key(doc: dict) -> str:
    """The ``chunk_id`` a kept row gets in ``sources`` (``<source>::<chunk>``)."""
    source = doc.get("source") or doc.get("filename") or "Unknown Document"
    raw_cid = doc.get("chunk_id")
    if raw_cid is None:
        raw_cid = doc.get("id")
    return f"{source}::{raw_cid}" if raw_cid is not None else f"{source}::{(doc.get('text') or '')[:64]}"


def _shadow_pool(vector_results: list, text_results: list) -> dict[str, str]:
    """Every candidate both channels returned, before the floors: key -> passage (plan S34)."""
    from core.knowledge.ranking.library_memory_shadow import library_passage

    pool: dict[str, str] = {}
    for doc in [*vector_results, *text_results]:
        key = _shadow_row_key(doc)
        if key not in pool:
            pool[key] = library_passage(
                str(doc.get("source") or doc.get("filename") or ""), str(doc.get("text") or "")
            )
    return pool


def _offer_shadow(
    shadow_path: str | None,
    query: str,
    pool: dict[str, str] | None,
    kept_keys: list,
    *,
    top_k: int,
    scoped: bool,
) -> None:
    if not shadow_path or not pool:
        return
    from core.knowledge.ranking.library_memory_shadow import offer

    rank = {key: i for i, key in enumerate(kept_keys)}
    rows = sorted(pool.items(), key=lambda kv: rank.get(kv[0], len(rank)))  # kept rows in rank order first
    offer(shadow_path, query, [(text, key in rank) for key, text in rows], top_k=top_k, scoped=scoped)


def rag_search(
    query: str,
    query_vector: np.ndarray,
    store: DocumentStore,
    top_k: int = 5,
    *,
    source_filter: str | None = None,
    source_prefix_filter: str | None = None,
    shadow_path: str | None = None,
    relevance_gate: bool = False,
) -> dict:
    """
    RAG v2.3 — Contract-safe retrieval system.

    Design goals:
    - No assumptions about DB hybrid capabilities
    - No brittle NLP preprocessing
    - No UI contract drift
    - Safe fallback behavior across all configurations
    - Strict RAM + context enforcement

    ``shadow_path`` (plan S34): during a worker turn, offer the candidate pool
    and the kept rows to the cross-encoder shadow pass under that path. It
    never changes what is kept.

    ``relevance_gate`` (plan S35, D55): on an unscoped search, the S3
    cross-encoder decides what is kept in place of the semantic floor, the
    FTS overlap check and the fusion order
    (``core.knowledge.ranking.library_memory_gate``). The result then carries
    ``sufficiency`` (``sufficient`` / ``partial`` / ``insufficient`` /
    ``none`` / ``unscored``) and ``relevance_gate`` (counts and scores, no
    text). Without the reranker, or on a scoped search, today's floors apply
    unchanged and ``sufficiency`` is ``unscored``.
    """
    gate_note: dict = {}
    out = _rag_search(
        query,
        query_vector,
        store,
        top_k,
        source_filter=source_filter,
        source_prefix_filter=source_prefix_filter,
        shadow_path=shadow_path,
        relevance_gate=relevance_gate,
        gate_note=gate_note,
    )
    if relevance_gate and "sufficiency" not in out:
        from core.knowledge.ranking.library_memory_gate import STATE_UNSCORED

        out["sufficiency"] = str(gate_note.get("state") or STATE_UNSCORED)
        out["relevance_gate"] = dict(gate_note) or {"mode": "gate", "state": STATE_UNSCORED}
    return out


def _build_rag_context(ordered_results: list, top_k: int) -> tuple[list[str], list[dict], int, list]:
    """Context blocks, UI sources, chars and the docs used, from ranked rows (section 3)."""
    context_blocks = []
    sources = []
    used = []

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
        used.append(doc)

    return context_blocks, sources, current_chars, used


def _gated_rag_search(
    query: str,
    vector_results: list,
    text_results: list,
    *,
    top_k: int,
    scoped: bool,
    shadow_path: str | None,
    gate_note: dict,
) -> dict | None:
    """Plan S35: keep the pool rows the cross-encoder scores >= keep, best first.

    ``None`` when there is nothing to judge or the reranker cannot run; the
    caller then applies today's floors (``gate_note`` says why).
    """
    from core.knowledge.ranking.library_memory_gate import (
        LIBRARY_THRESHOLDS,
        STATE_INSUFFICIENT,
        STATE_NONE,
        dropped_for_diag,
        gate_passages,
        kept_diag,
    )
    from core.knowledge.ranking.library_memory_shadow import collecting, library_passage, offer_scored
    from core.knowledge.ranking.sufficiency import assess_sufficiency

    pool: dict[str, dict] = {}
    for doc in [*vector_results, *text_results]:
        key = _shadow_row_key(doc)
        if key not in pool:
            pool[key] = doc
    if not pool:
        gate_note.update({"mode": "gate", "state": STATE_NONE})
        return None
    keys = list(pool)
    passages = [
        library_passage(str(pool[k].get("source") or pool[k].get("filename") or ""), str(pool[k].get("text") or ""))
        for k in keys
    ]
    gate = gate_passages(query, passages, LIBRARY_THRESHOLDS)
    if gate is None:
        from core.knowledge.ranking.library_memory_gate import skipped_diag

        gate_note.update(skipped_diag(query))
        return None

    ranked = [pool[keys[i]] for i in gate.kept_indices]
    context_blocks, sources, current_chars, used = _build_rag_context(ranked, top_k)
    score_of = {keys[i]: gate.scores[i] for i in range(len(gate.scores))}
    used_keys = [_shadow_row_key(d) for d in used]
    kept_scores = [score_of[k] for k in used_keys]
    state = assess_sufficiency(kept_scores, gate.thresholds).state.value if kept_scores else STATE_INSUFFICIENT
    diag = kept_diag(gate.diag, kept_scores)
    diag["state"] = state
    kept_set = set(gate.kept_indices)
    diag["rerank_dropped"] = dropped_for_diag(
        [
            (str(pool[keys[i]].get("source") or pool[keys[i]].get("filename") or ""), gate.scores[i])
            for i in range(len(gate.scores))
            if i not in kept_set
        ]
    )
    logger.info(
        "[RAG] relevance gate: state=%s kept=%d of %d (keep=%.1f, best=%s, %.0f ms)",
        state,
        len(sources),
        len(keys),
        gate.thresholds.keep,
        f"{max(gate.scores):.2f}" if gate.scores else "-",
        gate.latency_ms,
    )
    if shadow_path and collecting():
        used_set = set(used_keys)
        order = used_keys + [k for k in keys[: len(gate.scores)] if k not in used_set]
        idx = {k: i for i, k in enumerate(keys)}
        offer_scored(
            shadow_path,
            query,
            [(passages[idx[k]], k in used_set, gate.scores[idx[k]]) for k in order],
            top_k=top_k,
            scoped=scoped,
            latency_ms=gate.latency_ms,
        )
    if context_blocks:
        logger.info(f"[RAG v2.3] Returned {len(context_blocks)} chunks | chars={current_chars}")
    return {
        "llm_context": "\n\n".join(context_blocks),
        "sources": sources,
        "sufficiency": state,
        "relevance_gate": diag,
    }


def _rag_search(
    query: str,
    query_vector: np.ndarray,
    store: DocumentStore,
    top_k: int = 5,
    *,
    source_filter: str | None = None,
    source_prefix_filter: str | None = None,
    shadow_path: str | None = None,
    relevance_gate: bool = False,
    gate_note: dict,
) -> dict:
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
        gate_note.update({"mode": "gate", "state": "unscored", "skipped_reason": "reindex_in_progress"})
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
            # Unscoped search covers the user's documents only: help rows go
            # through @help, and memory rows through memory_search (plan V65).
            vector_results = _exclude_memory_results(_exclude_help_corpus_results(vector_results))
            text_results = _exclude_memory_results(_exclude_help_corpus_results(text_results))

        shadow_pool = None
        scoped = bool(source_filter or source_prefix_filter)
        if shadow_path:
            from core.knowledge.ranking.library_memory_shadow import collecting

            if collecting():
                shadow_pool = _shadow_pool(vector_results, text_results)

        # S35 (D55): the cross-encoder gate replaces the floors and the fusion
        # order below; without the reranker they apply unchanged.
        if relevance_gate:
            if scoped:
                gate_note.update({"mode": "gate", "state": "unscored", "skipped_reason": "scoped"})
            else:
                gated = _gated_rag_search(
                    query,
                    vector_results,
                    text_results,
                    top_k=top_k,
                    scoped=scoped,
                    shadow_path=shadow_path,
                    gate_note=gate_note,
                )
                if gated is not None:
                    return gated

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
            _offer_shadow(shadow_path, query, shadow_pool, [], top_k=top_k, scoped=scoped)
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
                _offer_shadow(shadow_path, query, shadow_pool, [], top_k=top_k, scoped=scoped)
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

        context_blocks, sources, current_chars, _used = _build_rag_context(ordered_results, top_k)

        # ============================================================
        # 4. FINAL RESPONSE
        # ============================================================

        logger.info(
            f"[RAG v2.3] Returned {len(context_blocks)} chunks | "
            f"chars={current_chars}"
        )
        _offer_shadow(
            shadow_path,
            query,
            shadow_pool,
            [s.get("chunk_id") for s in sources],
            top_k=top_k,
            scoped=scoped,
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