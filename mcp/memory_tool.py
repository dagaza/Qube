import logging
import json
import numpy as np

logger = logging.getLogger("Qube.MemoryTool")

MAX_MEMORY_CHARS = 2000
MAX_MEMORY_RESULTS = 5

# ============================================================
# v5.2 CHANGE: RELAXED + INTERPRETABLE SEMANTIC GATE
# (IMPORTANT: this is NOT a hard rejection anymore)
# ============================================================
MIN_SIMILARITY_DISTANCE = 0.75  # higher = more permissive retrieval (L2 distance)

# Candidate expansion factor (fixes starvation)
CANDIDATE_MULTIPLIER = 6

# Soft filtering (NOT hard cutoff)
SOFT_DISTANCE_CUTOFF = 0.85


def memory_search(query: str, query_vector: np.ndarray, store, top_k: int = 5, trace: bool = False) -> dict:
    """
    Memory v5.2 Retrieval Layer (Safe + Traceable)

    ✔ Fixes candidate starvation
    ✔ Uses soft ranking instead of hard rejection
    ✔ Adds trace mode for debugging retrieval decisions
    ✔ Stable for low-RAM local deployments
    """

    logger.info(f"[Memory v5.2] Searching: '{query}'")

    try:
        # ============================================================
        # 1. VECTOR SEARCH (EXPANDED CANDIDATES)
        # ============================================================
        results = (
            store.table
            .search(query_vector)
            .where("source LIKE 'qube_memory::%'")
            .limit(top_k * CANDIDATE_MULTIPLIER)
            .to_list()
        )

        if not results:
            return {"memory_context": "", "memory_sources": []}

        filtered = []

        # ============================================================
        # 2. SCORE + TRACE PIPELINE
        # ============================================================
        for r in results:
            distance = r.get("_distance", 1.0)

            try:
                payload = json.loads(r.get("text", "{}"))
            except Exception:
                if trace:
                    logger.debug(f"[Memory TRACE] JSON parse failed for row")
                continue

            content = payload.get("content", "")
            confidence = float(payload.get("confidence", 0.0))
            category = payload.get("category", "context")
            strength = payload.get("strength", 1)

            if not content:
                continue

            # ========================================================
            # v5.2 HYBRID SCORE (IMPORTANT UPGRADE)
            # ========================================================
            semantic_score = max(0.0, 1.0 - distance)
            final_score = (semantic_score * 0.65) + (confidence * 0.25) + (min(strength, 10) / 10 * 0.10)

            if trace:
                logger.debug(
                    f"[Memory TRACE] content='{content[:40]}...' "
                    f"dist={distance:.3f} "
                    f"semantic={semantic_score:.3f} "
                    f"confidence={confidence:.2f} "
                    f"strength={strength} "
                    f"score={final_score:.3f}"
                )

            # Soft gate (NOT a hard rejection anymore)
            if distance > SOFT_DISTANCE_CUTOFF:
                if trace:
                    logger.debug(f"[Memory TRACE] soft-rejected (too distant)")
                continue

            filtered.append({
                "content": content,
                "confidence": confidence,
                "distance": distance,
                "category": category,
                "strength": strength,
                "score": final_score
            })

        if not filtered:
            return {"memory_context": "", "memory_sources": []}

        # ============================================================
        # 3. SORT BY HYBRID SCORE
        # ============================================================
        filtered.sort(key=lambda x: x["score"], reverse=True)

        # ============================================================
        # 4. CONTEXT BUILD (STRICT BUDGET SAFE)
        # ============================================================
        context_blocks = []
        sources = []
        current_chars = 0

        for i, item in enumerate(filtered[:MAX_MEMORY_RESULTS], start=1):

            text = item["content"]
            if current_chars + len(text) > MAX_MEMORY_CHARS:
                break

            current_chars += len(text)

            context_blocks.append(f"- {text}")

            sources.append({
                "id": i,
                "filename": f"Memory: {item['category'].title()}",
                "content": text
            })

        logger.info(
            f"[Memory v5.2] returned={len(context_blocks)} "
            f"chars={current_chars} trace={trace}"
        )

        return {
            "memory_context": "\n".join(context_blocks),
            "memory_sources": sources
        }

    except Exception as e:
        logger.error(f"[Memory v5.2] search failed: {e}")
        return {"memory_context": "", "memory_sources": []}