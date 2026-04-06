# rag/chunker.py

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()
    
    if not text:
        return []

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break cleanly at natural boundaries
        if end < len(text):
            # 1. Priority: Look for a paragraph break
            split_idx = chunk.rfind("\n\n")
            
            # 2. Fallback: Look for a sentence boundary
            if split_idx < chunk_size // 2:
                split_idx = chunk.rfind(". ")
                
            # 3. Fallback: Simple line break (great for code, lists, and requirements.txt)
            if split_idx < chunk_size // 2:
                split_idx = chunk.rfind("\n")

            # If we found a good breaking point in the second half of the chunk, cut it there
            if split_idx > chunk_size // 2:
                end = start + split_idx + 1
                chunk = text[start:end]
                
        chunks.append(chunk.strip())
        start = end - overlap
        
    # Smart discard: Only throw away < 50 char chunks if they are useless tail fragments.
    # If the document is tiny and only generated 1 chunk, keep it!
    return [c for c in chunks if len(c) > 50 or len(chunks) == 1]