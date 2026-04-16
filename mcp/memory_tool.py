import logging
import json
import re
import numpy as np

from core.memory_usage_recorder import get_memory_usage_recorder

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

# ============================================================
# v6.1: HARD semantic-similarity floor.
# ------------------------------------------------------------
# Nomic v1.5 embeddings are L2-normalized, so semantic_score (1 - distance)
# is a proxy for cosine similarity. Anything below this floor is
# topically unrelated — injecting such rows into the LLM prompt is how a
# stored "my mom's name is Cornelia" memory ended up in a file-lookup
# query about "Dr. Evelyn", confusing the model into emitting a bare
# citation token. This is a HARD cutoff that runs *after* the soft
# distance gate so truly unrelated memories never reach the UI.
# ============================================================
MIN_SEMANTIC_SCORE = 0.35


# ============================================================
# v6.1: PROPER-NOUN KEYWORD-OVERLAP GATE.
# ------------------------------------------------------------
# When the user's query contains a distinctive proper noun (capitalized
# token that isn't a sentence-initial stopword — "Dr.", "Evelyn", "Omega",
# "Python", etc.), any memory candidate whose content does NOT mention at
# least one of those tokens is topically unrelated and must be dropped.
#
# Example this guards against:
#   Query:  "Tell me about Dr. Evelyn"        -> proper nouns: ["Dr", "Evelyn"]
#   Memory: "my mom's name is Cornelia"       -> overlap with proper nouns: 0
#   -> drop
#
# This gate is ONLY applied when the query actually contains proper nouns.
# Generic queries ("what are my preferences?", "summarize my notes") leave
# the permissive semantic-only gate in charge.
# ============================================================
_PROPER_NOUN_TOKEN_RE = re.compile(r"[A-Z][A-Za-z\-']{1,}")

# Proper nouns at sentence start / after punctuation that are almost always
# just English sentence starters — we don't want "Can", "Tell", "What",
# "Who", "Is", "Do", "Remember", etc. treated as "distinctive" entities.
_PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset(
    {
        "Can", "Could", "Would", "Should", "Shall", "Will", "Might", "May",
        "Do", "Does", "Did", "Is", "Are", "Was", "Were", "Be", "Been",
        "Have", "Has", "Had", "I", "You", "We", "They", "He", "She", "It",
        "My", "Your", "Our", "Their", "His", "Her", "Its",
        "Tell", "Show", "Give", "Please", "Remind", "Remember", "Memorize",
        "What", "Who", "Where", "When", "Why", "How", "Which",
        "Check", "Search", "Look", "Find", "Read", "Open", "Pull",
        "About", "Of", "To", "From", "In", "Into", "Inside", "Through",
        "On", "At", "For", "With", "Without", "By", "As",
        "The", "A", "An", "And", "Or", "But", "So", "Yet",
        "If", "Then", "Than", "That", "This", "These", "Those",
        # Domain-specific we do NOT want to count as a proper noun for
        # gate purposes (e.g. "Qube" is the app name, not a user entity).
        "Qube",
    }
)


def _extract_query_proper_nouns(query: str) -> list[str]:
    """Extract distinctive proper-noun tokens from the query.

    Returns tokens case-preserved. Sentence-initial stopwords (``Tell``,
    ``Who``, ``What``, ``My``, ``Can``, etc.) are excluded — they would
    otherwise trigger the overlap gate spuriously on every query.
    """
    if not query:
        return []
    # Strip everything but letters so "Dr." and "Dr.Evelyn" (no space) also
    # yield both "Dr" and "Evelyn".
    raw = _PROPER_NOUN_TOKEN_RE.findall(query)
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in _PROPER_NOUN_STOPWORDS:
            continue
        # Skip single-char tokens ("I").
        if len(t) < 2:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _content_has_any_token(content: str, tokens: list[str]) -> bool:
    """Case-insensitive whole-word presence check."""
    if not tokens or not content:
        return False
    c = content.lower()
    for t in tokens:
        t_low = t.lower()
        # Whole-word-ish match: token surrounded by non-word chars or string
        # boundary. Handles punctuation like "Dr." and possessive "'s".
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(t_low)}(?:$|[^a-z0-9])", c):
            return True
    return False

# ============================================================
# v6 PHASE B: link expansion budgets.
# When a returned memory carries ``links_to_document_ids`` we look up those
# RAG chunks and attach them as ``type="rag"`` sources alongside the memory
# entry. This is the mechanism that fixes the "person_name memory points
# back to the document chunk that describes them" failure.
# ============================================================
MAX_LINK_EXPANSIONS_PER_MEMORY = 2
MAX_LINK_EXPANSIONS_TOTAL = 4
MAX_LINK_CHARS_TOTAL = 2400


def _lookup_rag_chunk(store, link_id: str) -> dict | None:
    """Look a RAG chunk up by its composite ``"<source>::<chunk_int>"`` id.

    Returns a dict shaped like a rag_tool source (filename / content /
    chunk_id), or ``None`` if the chunk could not be located.
    """
    if not link_id or "::" not in link_id:
        return None
    src_name, _, raw_cid = link_id.rpartition("::")
    if not src_name or not raw_cid:
        return None
    try:
        cid_int = int(raw_cid)
    except (TypeError, ValueError):
        return None

    try:
        # Tantivy/Lance whereclause: escape single quotes by doubling.
        safe_src = src_name.replace("'", "''")
        rows = (
            store.table.search()
            .where(f"source = '{safe_src}' AND chunk_id = {cid_int}")
            .limit(1)
            .to_list()
        )
    except Exception as e:
        logger.debug(f"[Memory v6] link lookup failed for {link_id!r}: {e}")
        return None

    if not rows:
        return None
    row = rows[0]
    text = (row.get("text") or "").strip()
    if not text:
        return None
    return {
        "filename": row.get("source") or src_name,
        "content": text,
        "chunk_id": link_id,
    }


def memory_search(query: str, query_vector: np.ndarray, store, top_k: int = 5, trace: bool = False) -> dict:
    """
    Memory v6 Retrieval Layer (Safe + Traceable + Phase B link expansion)

    ✔ Fixes candidate starvation
    ✔ Uses soft ranking instead of hard rejection
    ✔ Adds trace mode for debugging retrieval decisions
    ✔ Stable for low-RAM local deployments
    ✔ Phase B: auto-expands ``links_to_document_ids`` into rag-type sources
      so thin memories surface their originating document chunk
    """

    logger.info(f"[Memory v6] Searching: '{query}'")

    # v6.1: collect distinctive proper-noun tokens from the query up-front.
    # When the query is entity-scoped (e.g. "tell me about Dr. Evelyn"),
    # memories that don't mention any of those tokens must be dropped —
    # otherwise an unrelated stored fact (e.g. "my mom's name is Cornelia")
    # can outrun the semantic-similarity floor on certain small embedders
    # and land in the LLM prompt as a false citation.
    query_proper_nouns = _extract_query_proper_nouns(query)

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
            decay = float(payload.get("decay", 1.0))
            links = payload.get("links_to_document_ids") or []

            if not content:
                continue

            # ========================================================
            # v6 HYBRID SCORE (Phase C: decay-weighted)
            # Re-weighted to include the decay term so memories that earn
            # their keep float to the top while stale rows sink without
            # being immediately purged.
            # ========================================================
            semantic_score = max(0.0, 1.0 - distance)
            final_score = (
                semantic_score * 0.55
                + confidence * 0.20
                + min(strength, 10) / 10 * 0.10
                + max(0.0, min(1.0, decay)) * 0.15
            )

            if trace:
                logger.debug(
                    f"[Memory TRACE] content='{content[:40]}...' "
                    f"dist={distance:.3f} "
                    f"semantic={semantic_score:.3f} "
                    f"confidence={confidence:.2f} "
                    f"strength={strength} "
                    f"score={final_score:.3f} "
                    f"links={len(links)}"
                )

            # Soft gate (NOT a hard rejection anymore)
            if distance > SOFT_DISTANCE_CUTOFF:
                if trace:
                    logger.debug(f"[Memory TRACE] soft-rejected (too distant)")
                continue

            # v6.1 HARD gate on semantic relevance. This catches the
            # "topically unrelated but not extremely distant" band where
            # the soft L2 gate still lets rows through (e.g. a personal
            # memory about a name scoring ~0.5 cosine against a generic
            # document-lookup query). Below the floor, the memory is
            # dropped entirely — no UI source, no LLM injection.
            if semantic_score < MIN_SEMANTIC_SCORE:
                if trace:
                    logger.debug(
                        f"[Memory TRACE] dropped (semantic={semantic_score:.3f} < {MIN_SEMANTIC_SCORE})"
                    )
                continue

            # v6.1 PROPER-NOUN GATE. When the query scopes to a specific
            # named entity (e.g. "tell me about Dr. Evelyn"), drop
            # candidates whose content has zero overlap with those
            # entities. This is what prevents a "my mom's name is
            # Cornelia" memory from surfacing on a Dr. Evelyn query.
            # Bypassed when the query has no distinctive proper nouns —
            # generic questions ("what are my preferences?", "summarize my
            # notes") still flow through the permissive semantic gate.
            if query_proper_nouns and not _content_has_any_token(
                content, query_proper_nouns
            ):
                if trace:
                    logger.debug(
                        "[Memory TRACE] dropped (no proper-noun overlap with %r)",
                        query_proper_nouns,
                    )
                continue

            filtered.append({
                "memory_id": r.get("id"),
                "content": content,
                "confidence": confidence,
                "distance": distance,
                "category": category,
                "strength": strength,
                "score": final_score,
                "links_to_document_ids": [str(x) for x in links if x],
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

        kept_items = []
        for item in filtered[:MAX_MEMORY_RESULTS]:
            text = item["content"]
            if current_chars + len(text) > MAX_MEMORY_CHARS:
                break
            current_chars += len(text)
            context_blocks.append(f"- {text}")
            kept_items.append(item)

        recorder = get_memory_usage_recorder()
        for i, item in enumerate(kept_items, start=1):
            mid = item.get("memory_id")
            sources.append({
                "id": i,
                "filename": f"Memory: {item['category'].title()}",
                "content": item["content"],
                "type": "memory",
                # Phase C additive: lance row id so llm_worker can attribute
                # citations back to the originating memory entry. Never
                # consumed by the UI rendering layer (which uses ``id``).
                "memory_id": mid,
            })
            if mid:
                recorder.record_retrieved(str(mid))

        # ============================================================
        # 5. PHASE B LINK EXPANSION
        #
        # For each kept memory with ``links_to_document_ids``, look the
        # linked rag chunks up and append them as additional rag-type
        # sources. The llm_worker's ``_apply_sequential_source_ids`` will
        # then assign globally sequential citation ids across the merged
        # list, so no downstream change is needed.
        # ============================================================
        expanded_links_seen: set[str] = set()
        link_chars = 0
        link_count = 0
        for item in kept_items:
            per_mem = 0
            for link_id in item.get("links_to_document_ids", []):
                if link_count >= MAX_LINK_EXPANSIONS_TOTAL:
                    break
                if per_mem >= MAX_LINK_EXPANSIONS_PER_MEMORY:
                    break
                if not link_id or link_id in expanded_links_seen:
                    continue
                chunk = _lookup_rag_chunk(store, link_id)
                if not chunk:
                    continue
                body = chunk["content"]
                if link_chars + len(body) > MAX_LINK_CHARS_TOTAL:
                    break
                link_chars += len(body)
                expanded_links_seen.add(link_id)
                per_mem += 1
                link_count += 1
                sources.append({
                    "id": len(sources) + 1,
                    "filename": chunk["filename"],
                    "content": body,
                    "type": "rag",
                    "chunk_id": chunk["chunk_id"],
                })

        if expanded_links_seen:
            logger.info(
                f"[Memory v6] expanded {len(expanded_links_seen)} memory link(s) into rag sources "
                f"({link_chars} chars)"
            )

        logger.info(
            f"[Memory v6] returned={len(context_blocks)} memory + "
            f"{len(expanded_links_seen)} link(s) "
            f"chars={current_chars} trace={trace}"
        )

        return {
            "memory_context": "\n".join(context_blocks),
            "memory_sources": sources,
        }

    except Exception as e:
        logger.error(f"[Memory v6] search failed: {e}")
        return {"memory_context": "", "memory_sources": []}
