import numpy as np
import logging

logger = logging.getLogger("Qube.Router")


class SemanticRetrievalRouter:
    """
    Qube Semantic Router v2

    ✔ Embedding-based intent classification
    ✔ Eliminates brittle keyword triggers
    ✔ Stable HYBRID detection
    ✔ Memory vs RAG vs Reasoning separation
    """

    def __init__(self, embedder):
        self.embedder = embedder

        # ------------------------------------------------------------
        # INTENT PROTOTYPES (semantic anchors)
        # ------------------------------------------------------------
        self.memory_anchor = self._embed(
            "personal memory recall about the user preferences past behavior facts about me"
        )

        self.rag_anchor = self._embed(
            "questions answered using documents files pdf notes sources or uploaded content"
        )

        self.reasoning_anchor = self._embed(
            "explanations comparisons analysis reasoning differences between concepts general knowledge question"
        )

    # ============================================================
    # EMBEDDING HELPER
    # ============================================================
    def _embed(self, text: str):
        try:
            return np.array(self.embedder.embed_query(text))
        except Exception as e:
            logger.error(f"[Router] embedding error: {e}")
            return np.zeros(768)

    # ============================================================
    # COSINE SIMILARITY
    # ============================================================
    def _cosine(self, a, b):
        if a is None or b is None:
            return 0.0

        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    # ============================================================
    # MAIN ROUTE FUNCTION
    # ============================================================
    def route(self, query: str, query_vector=None):

        q_emb = query_vector or self._embed(query)

        mem_score = self._cosine(q_emb, self.memory_anchor)
        rag_score = self._cosine(q_emb, self.rag_anchor)
        reasoning_score = self._cosine(q_emb, self.reasoning_anchor)

        logger.debug(
            f"[SemanticRouter] memory={mem_score:.3f} rag={rag_score:.3f} reasoning={reasoning_score:.3f}"
        )

        # ============================================================
        # DECISION LOGIC (stable + deterministic)
        # ============================================================

        # 1. MEMORY dominates if clearly personal
        if mem_score > 0.55 and mem_score > rag_score:
            return {
                "route": "memory",
                "confidence": mem_score,
                "memory_query": query,
                "rag_query": None,
                "strategy": "semantic memory recall"
            }

        # 2. RAG dominates if document grounded
        if rag_score > 0.55 and rag_score > mem_score:
            return {
                "route": "rag",
                "confidence": rag_score,
                "memory_query": None,
                "rag_query": query,
                "strategy": "semantic document retrieval"
            }

        # 3. HYBRID if both relevant
        if mem_score > 0.40 and rag_score > 0.40:
            return {
                "route": "hybrid",
                "confidence": max(mem_score, rag_score),
                "memory_query": query,
                "rag_query": query,
                "strategy": "semantic hybrid reasoning"
            }

        # 4. reasoning fallback (no retrieval)
        return {
            "route": "none",
            "confidence": reasoning_score,
            "memory_query": None,
            "rag_query": None,
            "strategy": "pure reasoning"
        }