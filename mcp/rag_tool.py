# mcp/rag_tool.py
from rag.store import DocumentStore
import logging
import numpy as np

logger = logging.getLogger("Qube.RAGTool")

# NOMIC TUNING: Nomic vectors behave slightly differently than MiniLM.
# 0.35 is a good starting point for Cosine similarity.
SIMILARITY_THRESHOLD = 350.0  

def rag_search(query: str, query_vector: np.ndarray, store: DocumentStore, top_k: int = 5) -> dict:
    """
    Executes a RAG search using a pre-computed vector.
    Returns a dictionary containing both the formatted context for the LLM prompt 
    and the structured metadata for the UI source chips.
    """
    logger.info(f"Executing RAG Search for: '{query}'")
    
    try:
        # THE HYBRID TRIGGER: Pass BOTH the meaning (vector) and the keywords (query)
        results = store.search(query_vector=query_vector, query_text=query, top_k=top_k)

        # UPDATE: Return the expected dictionary structure if empty
        if not results:
            logger.warning("RAG Search returned zero initial results.")
            return {"llm_context": "", "sources": []}

        # Filter out bad matches based on L2 distance
        filtered = []
        for r in results:
            distance = r.get("_distance", 1.0)
            logger.debug(f"Document: {r.get('source', 'Unknown')} | Distance: {distance:.4f}")
            
            # THE FIX: Reject anything with a distance higher than 350
            if distance < SIMILARITY_THRESHOLD:
                filtered.append(r)

        if not filtered:
            logger.info("Results found, but none passed the similarity threshold.")
            return {"llm_context": "", "sources": []}

        # THE PRESTIGE ARCHITECTURE: Construct dual payloads
        context_blocks = []
        source_metadata = []

        for i, r in enumerate(filtered, start=1):
            source_name = r.get('source', 'Unknown Document')
            text_chunk = r.get('text', '').strip()
            
            context_blocks.append(f"--- SOURCE {i}: {source_name} ---\n{text_chunk}")
            
            source_metadata.append({
                "id": i,
                "filename": source_name,
                "content": text_chunk
            })

        logger.info(f"Successfully retrieved {len(filtered)} relevant context blocks.")
        
        return {
            "llm_context": "\n\n".join(context_blocks),
            "sources": source_metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to execute RAG search: {e}")
        return {"llm_context": "", "sources": []}