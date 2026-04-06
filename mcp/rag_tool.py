# mcp/rag_tool.py
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore

# Tune this empirically. Lower = more permissive matching.
SIMILARITY_THRESHOLD = 0.35  

def rag_search(query: str, embedder: EmbeddingModel, store: DocumentStore, top_k: int = 5) -> str:
    query_vector = embedder.embed_one(query)
    results = store.search(query_vector, top_k=top_k)

    if not results:
        return ""

    # LanceDB returns _distance (L2). Convert to cosine-like score and threshold.
    filtered = [r for r in results if r.get("_distance", 1.0) < (1.0 - SIMILARITY_THRESHOLD)]

    if not filtered:
        return ""

    context_blocks = []
    for r in filtered:
        context_blocks.append(f"[Source: {r['source']}]\n{r['text']}")

    return "\n\n---\n\n".join(context_blocks)