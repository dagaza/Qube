# mcp/rag_tool.py
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
import logging

logger = logging.getLogger("Qube.RAGTool")

# 🔑 NOMIC TUNING: Nomic vectors behave slightly differently than MiniLM.
# 0.35 is a good starting point for Cosine similarity.
SIMILARITY_THRESHOLD = 350.0  

def rag_search(query: str, embedder: EmbeddingModel, store: DocumentStore, top_k: int = 5) -> dict:
    """
    Executes a RAG search and returns a dictionary containing both the formatted 
    context for the LLM prompt and the structured metadata for the UI source chips.
    """
    logger.info(f"Executing RAG Search for: '{query}'")
    
    try:
        # 1. Use the Nomic-specific query prefix
        query_vector = embedder.embed_query(query)
        
        # 2. Search the vector database
        results = store.search(query_vector, top_k=top_k)

        # 🔑 UPDATE: Return the expected dictionary structure if empty
        if not results:
            logger.warning("RAG Search returned zero initial results.")
            return {"llm_context": "", "sources": []}

        # 3. Filter out bad matches based on L2 distance
        filtered = []
        for r in results:
            distance = r.get("_distance", 1.0)
            logger.debug(f"Document: {r.get('source', 'Unknown')} | Distance: {distance:.4f}")
            
            # 🔑 THE FIX: Reject anything with a distance higher than 350
            if distance < SIMILARITY_THRESHOLD:
                filtered.append(r)

        if not filtered:
            logger.info("Results found, but none passed the similarity threshold.")
            return {"llm_context": "", "sources": []}

        # 4. 🔑 THE PRESTIGE ARCHITECTURE: Construct dual payloads
        context_blocks = []
        source_metadata = []

        # We use enumerate (1-indexed) to generate the [1], [2] citations
        for i, r in enumerate(filtered, start=1):
            source_name = r.get('source', 'Unknown Document')
            text_chunk = r.get('text', '').strip()
            
            # Payload A: For the LLM to read and cite (e.g., --- SOURCE 1 ---)
            context_blocks.append(f"--- SOURCE {i}: {source_name} ---\n{text_chunk}")
            
            # Payload B: For the UI to generate clickable chips
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