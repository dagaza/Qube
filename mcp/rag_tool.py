# mcp/rag_tool.py
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
import logging

logger = logging.getLogger("Qube.RAGTool")

# 🔑 NOMIC TUNING: Nomic vectors behave slightly differently than MiniLM.
# You may need to adjust this up or down based on your LanceDB metric (L2 vs Cosine).
# 0.35 is a good starting point for Cosine similarity.
SIMILARITY_THRESHOLD = 0.35  

def rag_search(query: str, embedder: EmbeddingModel, store: DocumentStore, top_k: int = 5) -> str:
    logger.info(f"Executing RAG Search for: '{query}'")
    
    try:
        # 1. Use the Nomic-specific query prefix
        query_vector = embedder.embed_query(query)
        
        # 2. Search the vector database
        results = store.search(query_vector, top_k=top_k)

        if not results:
            logger.warning("RAG Search returned zero initial results.")
            return ""

        # 3. Filter out bad matches based on distance
        filtered = []
        for r in results:
            # Safely grab the distance, defaulting to 1.0 (a bad match) if missing
            distance = r.get("_distance", 1.0)
            
            # If distance is less than our threshold curve, it's a good match!
            if distance < (1.0 - SIMILARITY_THRESHOLD):
                filtered.append(r)
            
            # Optional: Log the distances while you are testing so you know how to tune the threshold
            logger.debug(f"Document: {r.get('source', 'Unknown')} | Distance: {distance:.4f}")

        if not filtered:
            logger.info("Results found, but none passed the similarity threshold.")
            return ""

        # 4. Construct the final context block for the LLM
        context_blocks = []
        for r in filtered:
            # We wrap it in clear markers so the LLM knows exactly where the info came from
            source_name = r.get('source', 'Unknown Document')
            text_chunk = r.get('text', '').strip()
            
            context_blocks.append(f"--- START SOURCE: {source_name} ---\n{text_chunk}\n--- END SOURCE ---")

        logger.info(f"Successfully retrieved {len(filtered)} relevant context blocks.")
        return "\n\n".join(context_blocks)
        
    except Exception as e:
        logger.error(f"Failed to execute RAG search: {e}")
        return ""