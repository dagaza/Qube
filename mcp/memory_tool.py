import logging
import json
import re
import time
import numpy as np

from core.lance_row_id import LANCE_ROW_ID_SELECT, lance_row_id
from core.memory_filters import is_memory_actionable
# The policy functions are aliased: ``memory_search`` takes boolean
# arguments named ``apply_core_memory_gate`` and ``apply_mmr``, which would
# otherwise shadow them inside the function (plan V66).
from core.memory_retrieval_policy import (
    HYBRID_CANDIDATE_MULTIPLIER,
    apply_core_memory_gate as _policy_core_memory_gate,
    apply_mmr as _policy_mmr,
    fts_query_token_overlap,
    temporal_decay_multiplier,
    tier_from_source,
)
from core.memory_usage_recorder import get_memory_usage_recorder
from core.retrieval_fusion import (
    _fts_has_score_metadata,
    fuse_ranked_results,
    fuse_weighted_scores,
)
from core.reindex_state import is_reindex_in_progress

logger = logging.getLogger("Qube.MemoryTool")

MAX_MEMORY_CHARS = 2000
MAX_MEMORY_RESULTS = 5

# ============================================================
# v5.2 CHANGE: RELAXED + INTERPRETABLE SEMANTIC GATE
# (IMPORTANT: this is NOT a hard rejection anymore)
# ============================================================
MIN_SIMILARITY_DISTANCE = 0.75  # higher = more permissive retrieval (L2 distance)

# Candidate expansion factor (fixes starvation); v7 hybrid uses 4 (default multiplier).
CANDIDATE_MULTIPLIER = HYBRID_CANDIDATE_MULTIPLIER

# Soft filtering (NOT hard cutoff)
SOFT_DISTANCE_CUTOFF = 0.85

# ============================================================
# v6.1: HARD semantic-similarity floor.
# ------------------------------------------------------------
# L2-normalized fastembed vectors — semantic_score (1 - distance) is a cosine proxy.
# Anything below this floor is topically unrelated — injecting such rows into the LLM prompt is how a
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


_PROPER_NOUN_TITLES: frozenset[str] = frozenset({"Dr", "Mr", "Mrs", "Ms", "Mx", "Prof", "St"})
_PROPER_NOUN_CLITIC_RE = re.compile(r"(?:'s|'re|'ve|'ll|'d|'m|n't|')$")
_PROPER_NOUN_PREV_WORD_RE = re.compile(r"[A-Za-z]+\.?$")


def _starts_sentence(before: str) -> bool:
    """True when the token after ``before`` (already right-stripped) opens a sentence."""
    if not before:
        return True
    if before[-1] not in ".!?":
        return False
    prev = _PROPER_NOUN_PREV_WORD_RE.findall(before)
    return not (prev and prev[0].rstrip(".") in _PROPER_NOUN_TITLES)  # "Dr. Evelyn"


def _extract_query_proper_nouns(query: str) -> list[str]:
    """Extract distinctive proper-noun tokens from the query.

    Returns tokens case-preserved, with possessives and contractions stripped
    ("Kofi's" -> "Kofi"). Skipped (plan V67, D46):

    * the first word of each sentence: its capital says nothing, and sentence
      starters ("What's", "Am", "Write", "Help", "Explain") used to count as
      names and drop every memory;
    * titles ("Dr", "Mrs", ...): "Dr. Evelyn" requires "Evelyn";
    * the stopwords below, single characters and repeats.

    A name that opens the question ("Kofi's birthday?") is therefore not
    required; the semantic floor still applies.
    """
    if not query:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _PROPER_NOUN_TOKEN_RE.finditer(query):
        token = _PROPER_NOUN_CLITIC_RE.sub("", match.group(0))
        if _starts_sentence(query[: match.start()].rstrip()):
            continue
        if token in _PROPER_NOUN_TITLES or token in _PROPER_NOUN_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
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


_TIER_TO_SOURCE_PREFIX: dict[str, str] = {
    "preference": "qube_memory::preference::%",
    "knowledge": "qube_memory::knowledge::%",
    "episode": "qube_memory::episode::%",
    "context": "qube_memory::context::%",
    # Pre-T3.4 rows are migrated once to ``qube_memory::legacy::<category>``
    # (see ``core.memory_source_migration``). Chat/context retrieval pulls
    # this explicit namespace — never a catch-all ``qube_memory::%`` that
    # would defeat tier isolation for knowledge/episode rows.
    "legacy": "qube_memory::legacy::%",
}


def _build_tier_where_clause(
    *,
    include_preference: bool,
    include_knowledge: bool,
    include_episode: bool,
    include_context: bool,
) -> str:
    """Compose the LanceDB WHERE clause from the tier flags.

    Returns ``"source LIKE 'qube_memory::%'"`` when every tier flag is
    disabled (defensive — avoids an empty WHERE that would drop all
    rows). The legacy ``qube_memory::<category>`` namespace is routed
    together with ``context`` so pre-T3.4 memories keep surfacing on
    plain chat turns.
    """
    prefixes: list[str] = []
    if include_preference:
        prefixes.append(_TIER_TO_SOURCE_PREFIX["preference"])
    if include_knowledge:
        prefixes.append(_TIER_TO_SOURCE_PREFIX["knowledge"])
    if include_episode:
        prefixes.append(_TIER_TO_SOURCE_PREFIX["episode"])
    if include_context:
        prefixes.append(_TIER_TO_SOURCE_PREFIX["context"])
        prefixes.append(_TIER_TO_SOURCE_PREFIX["legacy"])

    if not prefixes:
        return "source LIKE 'qube_memory::%'"

    # Dedup while preserving order (legacy + context differ in prefix
    # but may still overlap logically if we ever rename the table).
    seen: set[str] = set()
    uniq: list[str] = []
    for p in prefixes:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return " OR ".join(f"source LIKE '{p}'" for p in uniq)


def _offer_memory_shadow(
    shadow_path: str | None,
    query: str,
    pool: dict[str, str] | None,
    kept_keys: list[str],
    *,
    top_k: int,
) -> None:
    if not shadow_path or not pool:
        return
    from core.knowledge.ranking.library_memory_shadow import offer

    rank = {key: i for i, key in enumerate(kept_keys)}
    rows = sorted(pool.items(), key=lambda kv: rank.get(kv[0], len(rank)))  # kept rows in rank order first
    # ``top_k`` sizes the candidate fetch; every mode cites up to MAX_MEMORY_RESULTS.
    offer(
        shadow_path,
        query,
        [(text, key in rank) for key, text in rows],
        top_k=min(int(top_k), MAX_MEMORY_RESULTS) if top_k else MAX_MEMORY_RESULTS,
    )


def _gated_memory_result(
    query: str,
    gate_pool: dict,
    gate,
    kept_items: list[dict],
    *,
    shadow_path: str | None,
    top_k: int,
    extra: dict,
) -> dict:
    """Plan S35: the gated recall result, its state and diag, and the shadow offer."""
    from core.knowledge.ranking.library_memory_gate import STATE_INSUFFICIENT, kept_diag
    from core.knowledge.ranking.library_memory_shadow import collecting, offer_scored
    from core.knowledge.ranking.sufficiency import assess_sufficiency

    keys = list(gate_pool)
    score_of = {keys[i]: gate.scores[i] for i in range(len(gate.scores))}
    kept_keys = [str(item.get("memory_id") or item["content"]) for item in kept_items]
    kept_scores = [score_of[k] for k in kept_keys if k in score_of]
    state = assess_sufficiency(kept_scores, gate.thresholds).state.value if kept_scores else STATE_INSUFFICIENT
    diag = kept_diag(gate.diag, kept_scores)
    diag["state"] = state
    logger.info(
        "[Memory v7] relevance gate: state=%s kept=%d of %d (keep=%.1f, best=%s, %.0f ms)",
        state,
        len(kept_items),
        len(keys),
        gate.thresholds.keep,
        f"{max(gate.scores):.2f}" if gate.scores else "-",
        gate.latency_ms,
    )
    if shadow_path and collecting():
        kept_set = set(kept_keys)
        order = kept_keys + [k for k in keys[: len(gate.scores)] if k not in kept_set]
        offer_scored(
            shadow_path,
            query,
            [(gate_pool[k][0], k in kept_set, score_of[k]) for k in order if k in score_of],
            top_k=min(int(top_k), MAX_MEMORY_RESULTS) if top_k else MAX_MEMORY_RESULTS,
            latency_ms=gate.latency_ms,
        )
    out = {"memory_context": "", "memory_sources": []}
    out.update(extra)
    out["sufficiency"] = state
    out["relevance_gate"] = diag
    return out


def memory_search(
    query: str,
    query_vector: np.ndarray,
    store,
    top_k: int = 5,
    trace: bool = False,
    prefer_episode: bool = False,
    *,
    include_preference: bool = True,
    include_knowledge: bool = False,
    include_episode: bool = False,
    include_context: bool = True,
    apply_core_memory_gate: bool = False,
    core_memory_min_score: float | None = None,
    core_memory_min_margin: float | None = None,
    apply_mmr: bool = False,
    apply_temporal_decay: bool = False,
    query_fingerprint: str | None = None,
    exclude_presentation_preferences: bool = False,
    shadow_path: str | None = None,
    relevance_gate: bool = False,
) -> dict:
    """
    Memory v6 Retrieval Layer (Safe + Traceable + Phase B link expansion)

    ✔ Fixes candidate starvation
    ✔ Uses soft ranking instead of hard rejection
    ✔ Adds trace mode for debugging retrieval decisions
    ✔ Stable for low-RAM local deployments
    ✔ Phase B: auto-expands ``links_to_document_ids`` into rag-type sources
      so thin memories surface their originating document chunk
    ✔ T3.2: ``prefer_episode=True`` boosts ``category=="episode"`` rows
      above atomic facts AND bypasses the proper-noun gate for them so
      narrative recap queries ("what have we been working on?") surface
      the session summary even when the query has no named entity.
    ✔ T3.4: tier flags (``include_preference`` / ``include_knowledge`` /
      ``include_episode`` / ``include_context``) scope the LanceDB WHERE
      clause so plain chat turns see only preferences + context while
      recall / hybrid / narrative turns also see knowledge and episodes.
    ✔ S34: ``shadow_path`` offers the candidate pool (after the tier filter and
      the action-boundary check, before the floors and gates) and the kept
      memories to the cross-encoder shadow pass during a worker turn. It
      never changes what is kept.
    ✔ S35 (D55, D56): ``relevance_gate`` on a recall search (no core-memory
      gate, no episode preference) lets the S3 cross-encoder decide what is
      kept in place of the distance cutoff, the semantic floor, the FTS
      overlap check, the score order and MMR. Memories are scored in the
      first person. The proper-noun check stays, on the kept rows. The
      result carries ``sufficiency`` and ``relevance_gate`` (counts and
      scores, no text). Without the reranker today's path applies unchanged.
    """
    gate_note: dict = {}
    out = _memory_search(
        query,
        query_vector,
        store,
        top_k,
        trace,
        prefer_episode,
        include_preference=include_preference,
        include_knowledge=include_knowledge,
        include_episode=include_episode,
        include_context=include_context,
        apply_core_memory_gate=apply_core_memory_gate,
        core_memory_min_score=core_memory_min_score,
        core_memory_min_margin=core_memory_min_margin,
        apply_mmr=apply_mmr,
        apply_temporal_decay=apply_temporal_decay,
        query_fingerprint=query_fingerprint,
        exclude_presentation_preferences=exclude_presentation_preferences,
        shadow_path=shadow_path,
        relevance_gate=relevance_gate,
        gate_note=gate_note,
    )
    if relevance_gate and "sufficiency" not in out:
        from core.knowledge.ranking.library_memory_gate import STATE_UNSCORED

        out["sufficiency"] = str(gate_note.get("state") or STATE_UNSCORED)
        out["relevance_gate"] = dict(gate_note) or {"mode": "gate", "state": STATE_UNSCORED}
    return out


def _memory_search(
    query: str,
    query_vector: np.ndarray,
    store,
    top_k: int = 5,
    trace: bool = False,
    prefer_episode: bool = False,
    *,
    include_preference: bool = True,
    include_knowledge: bool = False,
    include_episode: bool = False,
    include_context: bool = True,
    apply_core_memory_gate: bool = False,
    core_memory_min_score: float | None = None,
    core_memory_min_margin: float | None = None,
    apply_mmr: bool = False,
    apply_temporal_decay: bool = False,
    query_fingerprint: str | None = None,
    exclude_presentation_preferences: bool = False,
    shadow_path: str | None = None,
    relevance_gate: bool = False,
    gate_note: dict,
) -> dict:

    if is_reindex_in_progress():
        logger.info("[Memory] Retrieval suppressed while library reprocessing is active.")
        gate_note.update({"mode": "gate", "state": "unscored", "skipped_reason": "reindex_in_progress"})
        return {"memory_context": "", "memory_sources": []}

    logger.info(
        "[Memory v6] Searching: %r (prefer_episode=%s pref=%s know=%s ep=%s ctx=%s)",
        query,
        prefer_episode,
        include_preference,
        include_knowledge,
        include_episode,
        include_context,
    )
    where_clause = _build_tier_where_clause(
        include_preference=include_preference,
        include_knowledge=include_knowledge,
        include_episode=include_episode or prefer_episode,
        include_context=include_context,
    )

    # v6.1: collect distinctive proper-noun tokens from the query up-front.
    # When the query is entity-scoped (e.g. "tell me about Dr. Evelyn"),
    # memories that don't mention any of those tokens must be dropped —
    # otherwise an unrelated stored fact (e.g. "my mom's name is Cornelia")
    # can outrun the semantic-similarity floor on certain small embedders
    # and land in the LLM prompt as a false citation.
    query_proper_nouns = _extract_query_proper_nouns(query)

    try:
        limit = top_k * CANDIDATE_MULTIPLIER

        # ============================================================
        # 1. VECTOR + FTS CHANNELS (v7 hybrid)
        # ============================================================
        vector_results: list = []
        try:
            vector_results = (
                store.table
                .search(query_vector)
                .select(LANCE_ROW_ID_SELECT)
                .where(where_clause)
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.debug("[Memory v7] vector search failed: %s", e)

        text_results: list = []
        try:
            text_results = (
                store.table.search(query, query_type="fts")
                .select(["_rowid", "text", "source", "chunk_id"])
                .where(where_clause)
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.debug("[Memory v7] FTS search unavailable: %s", e)

        if not vector_results and not text_results:
            gate_note.update({"mode": "gate", "state": "none"})
            return {"memory_context": "", "memory_sources": []}

        def _mem_doc_id(doc: dict) -> str:
            return lance_row_id(doc) or str(doc.get("source") or (doc.get("text") or "")[:64])

        if _fts_has_score_metadata(text_results):
            fused = fuse_weighted_scores(
                vector_results,
                text_results,
                doc_id_fn=_mem_doc_id,
            )
        else:
            fused = fuse_ranked_results(vector_results, text_results, doc_id_fn=_mem_doc_id)
        if not fused:
            gate_note.update({"mode": "gate", "state": "none"})
            return {"memory_context": "", "memory_sources": []}

        filtered = []
        now_ts = time.time()
        shadow_pool: dict[str, str] | None = None
        if shadow_path:
            from core.knowledge.ranking.library_memory_shadow import collecting

            if collecting():
                shadow_pool = {}
        # S35: the gate's pool (every candidate before the floors, as the shadow
        # pass sees it), only on recall searches.
        gate_pool: dict[str, tuple[str, dict]] | None = None
        if relevance_gate:
            if apply_core_memory_gate or prefer_episode:
                gate_note.update({"mode": "gate", "state": "unscored", "skipped_reason": "not_recall"})
            else:
                gate_pool = {}

        # ============================================================
        # 2. SCORE + TRACE PIPELINE
        # ============================================================
        for r, channels in fused:
            distance = r.get("_distance", 1.0)
            semantic_score = max(0.0, 1.0 - float(distance))

            try:
                payload = json.loads(r.get("text", "{}"))
            except Exception:
                if trace:
                    logger.debug("[Memory TRACE] JSON parse failed for row")
                continue

            if not is_memory_actionable(payload, now=now_ts):
                if trace:
                    logger.debug("[Memory TRACE] dropped (action boundary expired)")
                continue

            if exclude_presentation_preferences and apply_core_memory_gate:
                kind = str(payload.get("preference_kind") or "").lower()
                if kind == "presentation":
                    if trace:
                        logger.debug("[Memory TRACE] dropped (presentation; policy layer)")
                    continue

            content = payload.get("content", "")
            confidence = float(payload.get("confidence", 0.0))
            category = payload.get("category", "context")
            strength = payload.get("strength", 1)
            decay = float(payload.get("decay", 1.0))
            links = payload.get("links_to_document_ids") or []
            source = r.get("source") or ""

            if not content:
                continue

            if shadow_pool is not None:
                from core.knowledge.ranking.library_memory_shadow import memory_passage

                shadow_pool.setdefault(str(lance_row_id(r) or content), memory_passage(content))

            has_vector = "vector" in channels
            has_fts = "fts" in channels
            vector_reliable = (
                has_vector
                and float(distance) <= SOFT_DISTANCE_CUTOFF
                and semantic_score >= MIN_SEMANTIC_SCORE
            )
            fts_only = has_fts and not vector_reliable

            if gate_pool is not None:
                from core.knowledge.ranking.library_memory_shadow import memory_passage

                gate_key = str(lance_row_id(r) or content)
                if gate_key not in gate_pool:
                    gate_score = (
                        semantic_score * 0.55
                        + confidence * 0.20
                        + min(strength, 10) / 10 * 0.10
                        + max(0.0, min(1.0, decay)) * 0.15
                    )
                    if fts_only:
                        gate_score = max(gate_score, confidence * 0.35 + 0.25)
                    if apply_temporal_decay:
                        gate_score *= temporal_decay_multiplier(tier_from_source(str(source)), payload, now=now_ts)
                    gate_pool[gate_key] = (
                        memory_passage(content),
                        {
                            "memory_id": lance_row_id(r),
                            "content": content,
                            "confidence": confidence,
                            "distance": distance,
                            "category": category,
                            "strength": strength,
                            "score": gate_score,
                            "links_to_document_ids": [str(x) for x in links if x],
                            "_row": r,
                            # The proper-noun check stays under the gate, on the kept rows.
                            "_proper_noun_ok": not (
                                query_proper_nouns
                                and not fts_only
                                and not _content_has_any_token(content, query_proper_nouns)
                            ),
                        },
                    )

            if fts_only and not fts_query_token_overlap(query, content):
                if trace:
                    logger.debug("[Memory TRACE] dropped (FTS-only, no token overlap)")
                continue

            if not fts_only and not vector_reliable:
                if trace:
                    logger.debug("[Memory TRACE] dropped (weak vector channel)")
                continue

            final_score = (
                semantic_score * 0.55
                + confidence * 0.20
                + min(strength, 10) / 10 * 0.10
                + max(0.0, min(1.0, decay)) * 0.15
            )
            if fts_only:
                final_score = max(final_score, confidence * 0.35 + 0.25)

            if apply_temporal_decay:
                tier = tier_from_source(str(source))
                final_score *= temporal_decay_multiplier(tier, payload, now=now_ts)

            is_episode = (category == "episode")
            if prefer_episode and is_episode:
                final_score += 0.35

            if trace:
                logger.debug(
                    f"[Memory TRACE] content='{content[:40]}...' "
                    f"channels={channels} dist={distance:.3f} "
                    f"semantic={semantic_score:.3f} score={final_score:.3f}"
                )

            if not fts_only:
                if float(distance) > SOFT_DISTANCE_CUTOFF:
                    if trace:
                        logger.debug("[Memory TRACE] soft-rejected (too distant)")
                    continue
                if semantic_score < MIN_SEMANTIC_SCORE:
                    if trace:
                        logger.debug(
                            "[Memory TRACE] dropped (semantic=%.3f < %.2f)",
                            semantic_score,
                            MIN_SEMANTIC_SCORE,
                        )
                    continue

            if (
                query_proper_nouns
                and not fts_only
                and not _content_has_any_token(content, query_proper_nouns)
                and not (prefer_episode and is_episode)
            ):
                if trace:
                    logger.debug(
                        "[Memory TRACE] dropped (no proper-noun overlap with %r)",
                        query_proper_nouns,
                    )
                continue

            filtered.append({
                "memory_id": lance_row_id(r),
                "content": content,
                "confidence": confidence,
                "distance": distance,
                "category": category,
                "strength": strength,
                "score": final_score,
                "links_to_document_ids": [str(x) for x in links if x],
                "_row": r,
            })

        gate = None
        if gate_pool:
            from core.knowledge.ranking.library_memory_gate import (
                MEMORY_RECALL_THRESHOLDS,
                gate_passages,
                skipped_diag,
            )

            gate_keys = list(gate_pool)
            gate = gate_passages(query, [gate_pool[k][0] for k in gate_keys], MEMORY_RECALL_THRESHOLDS)
            if gate is None:
                gate_note.update(skipped_diag(query))
            else:
                filtered = [
                    gate_pool[gate_keys[i]][1]
                    for i in gate.kept_indices
                    if gate_pool[gate_keys[i]][1]["_proper_noun_ok"]
                ]
        elif gate_pool is not None:
            gate_note.update({"mode": "gate", "state": "none"})

        if not filtered:
            if gate is not None:
                return _gated_memory_result(
                    query, gate_pool, gate, [], shadow_path=shadow_path, top_k=top_k, extra={}
                )
            _offer_memory_shadow(shadow_path, query, shadow_pool, [], top_k=top_k)
            return {"memory_context": "", "memory_sources": []}

        if gate is None:
            filtered.sort(key=lambda x: x["score"], reverse=True)

        if apply_mmr and gate is None:
            filtered = _policy_mmr(filtered, top_k=MAX_MEMORY_RESULTS)

        if apply_core_memory_gate:
            gated = _policy_core_memory_gate(
                filtered,
                min_score=core_memory_min_score,
                min_margin=core_memory_min_margin,
            )
            if not gated:
                logger.info("[Memory v7] core memory suppressed (relevance gate)")
                _offer_memory_shadow(shadow_path, query, shadow_pool, [], top_k=top_k)
                return {"memory_context": "", "memory_sources": []}
            filtered = gated

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
            # T3.2: label episode rows inline so the LLM can tell a
            # session summary apart from an atomic fact without extra
            # plumbing. NARRATIVE_RECALL_SYSTEM_SUFFIX explicitly tells
            # the LLM to prefer these rows on recap turns.
            label = "EPISODE" if item.get("category") == "episode" else None
            if label:
                context_blocks.append(f"- [{label}] {text}")
            else:
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
                recorder.record_retrieved(
                    str(mid),
                    query_fingerprint=query_fingerprint,
                    retrieval_score=float(item.get("score") or 0.0),
                )

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
        if gate is not None:
            return _gated_memory_result(
                query,
                gate_pool,
                gate,
                kept_items,
                shadow_path=shadow_path,
                top_k=top_k,
                extra={"memory_context": "\n".join(context_blocks), "memory_sources": sources},
            )
        _offer_memory_shadow(
            shadow_path,
            query,
            shadow_pool,
            [str(item.get("memory_id") or item["content"]) for item in kept_items],
            top_k=top_k,
        )

        return {
            "memory_context": "\n".join(context_blocks),
            "memory_sources": sources,
        }

    except Exception as e:
        logger.error(f"[Memory v6] search failed: {e}")
        return {"memory_context": "", "memory_sources": []}
