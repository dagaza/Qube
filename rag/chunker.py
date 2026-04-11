# rag/chunker.py
"""Document chunking for RAG ingestion — every segment is hard-capped in length."""

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, preferring natural breaks.
    Falls back to a brute-force slice so no single chunk exceeds chunk_size characters.
    """
    chunks: list[str] = []
    text = text.strip()
    if not text:
        return []

    overlap = max(0, min(overlap, max(1, chunk_size - 1)))
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        if end < n:
            split_idx = chunk.rfind("\n\n")
            if split_idx < chunk_size // 2:
                split_idx = chunk.rfind(". ")
            if split_idx < chunk_size // 2:
                split_idx = chunk.rfind("\n")
            if split_idx > chunk_size // 2:
                end = start + split_idx + 1
                chunk = text[start:end]

        # Hard cap — protects against pathological splits / huge single-line tables
        if len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]
            end = start + len(chunk)

        stripped = chunk.strip()
        if not stripped:
            start = end
            continue

        if len(stripped) > chunk_size:
            stripped = stripped[:chunk_size]
        chunks.append(stripped)

        if end >= n:
            break
        start = end - overlap

    return [c for c in chunks if len(c) > 50 or len(chunks) == 1]
