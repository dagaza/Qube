from rag.store import DocumentStore
import logging
import numpy as np

logger = logging.getLogger("Qube.RAGTool")

MAX_CONTEXT_CHARS = 12000


def rag_search(query: str, query_vector: np.ndarray, store: DocumentStore, top_k: int = 5) -> dict:
    """
    RAG v2.3 — Contract-safe retrieval system.

    Design goals:
    - No assumptions about DB hybrid capabilities
    - No brittle NLP preprocessing
    - No UI contract drift
    - Safe fallback behavior across all configurations
    - Strict RAM + context enforcement
    """

    logger.info(f"[RAG v2.3] Query: {query}")

    try:
        # ============================================================
        # 1. RETRIEVAL CONTRACT LAYER (SAFE & EXPLICIT)
        # ============================================================

        vector_results = []
        text_results = []

        # --- VECTOR SEARCH (semantic channel) ---
        try:
            vector_results = (
                store.table.search(query_vector)
                .limit(top_k * 2)
                .to_list()
            )
        except Exception as e:
            logger.error(f"[RAG] Vector search failed: {e}")

        # --- TEXT SEARCH (lexical fallback, optional capability) ---
        try:
            # NOTE:
            # Some setups require query_type="fts" to enable BM25.
            # If unsupported, this will safely fail and be ignored.
            text_results = (
                store.table.search(query, query_type="fts")
                .limit(top_k * 2)
                .to_list()
            )
        except Exception as e:
            logger.debug(f"[RAG] FTS search unavailable: {e}")

        # If everything fails, return safely
        if not vector_results and not text_results:
            logger.warning("[RAG] No retrieval results from any channel.")
            return {
                "llm_context": "",
                "sources": []
            }

        # ============================================================
        # 2. SAFE FUSION LAYER (DB-AGNOSTIC RANK MERGE)
        # ============================================================

        fused_scores = {}
        doc_map = {}

        def add_results(results, weight: float):
            for rank, doc in enumerate(results):
                doc_id = (
                    doc.get("chunk_id")
                    or doc.get("id")
                    or doc.get("source")
                    or doc.get("text", "")[:64]
                )

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_map[doc_id] = doc

                # rank-based scoring (stable across DBs)
                fused_scores[doc_id] += weight / (rank + 1)

        # Vector is primary signal
        add_results(vector_results, weight=1.0)

        # Text is fallback signal
        add_results(text_results, weight=0.8)

        ranked_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ordered_results = [doc_map[doc_id] for doc_id, _ in ranked_docs]

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

            # LLM context block
            context_blocks.append(
                f"--- SOURCE {i}: {source} ---\n{text}"
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