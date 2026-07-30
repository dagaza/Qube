"""MMR reranking for Library RAG hits (post-fusion deduplication)."""

from __future__ import annotations

from core.memory_retrieval_policy import apply_mmr

# Match web section ranker default (``section_ranker.MMR_LAMBDA``).
RAG_MMR_LAMBDA = 0.72


def apply_mmr_to_rag_hits(
    docs: list[dict],
    *,
    top_k: int,
    lambda_: float = RAG_MMR_LAMBDA,
) -> list[dict]:
    """
    Rerank fused RAG rows with MMR on chunk body text.

    Fusion rank is used as the relevance proxy (``1 / (rank + 1)``).
    """
    if not docs:
        return []
    if len(docs) <= 1:
        return docs[:top_k]

    mmr_items = []
    for rank, doc in enumerate(docs):
        mmr_items.append(
            {
                "score": 1.0 / (rank + 1),
                "content": (doc.get("text") or "").strip(),
                "_doc": doc,
            }
        )

    selected = apply_mmr(mmr_items, lambda_=lambda_, top_k=top_k)
    return [item["_doc"] for item in selected]
