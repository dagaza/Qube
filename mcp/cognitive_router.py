import time
import logging
from collections import deque, defaultdict
import numpy as np

from core.memory_filters import detect_recall_intent

logger = logging.getLogger("Qube.CognitiveRouterV4")


# ============================================================
# Substring trigger lists (module-level so tests can introspect
# them via len() and so we don't re-allocate on every route call).
# ============================================================
_MEMORY_TRIGGERS: tuple[str, ...] = (
    "remember", "did i", "do i like", "my preference",
    "what am i", "about me", "my settings",
)

_RAG_TRIGGERS: tuple[str, ...] = (
    "pdf", "document", "file", "notes",
    "according to", "in my", "library",
)

_WEB_TRIGGERS: tuple[str, ...] = (
    "look online", "search the web", "find on the internet",
    "google", "check online", "web search", "news", "current",
    "latest", "today", "right now", "happening", "weather",
)


# ============================================================
# Tier-2 confidence layer constants.
#
# - MIN_CONFIDENCE_FLOOR: ``top_score`` below this means no lane has
#   a real signal. Acts as a SECONDARY guard on top of the existing
#   per-lane dynamic thresholds — the downgrade only fires when the
#   top score is BOTH below this floor AND failed to clear its own
#   lane threshold by more than 0.05 (see route() for the exact
#   condition). This anchoring prevents embeddings from silently
#   overriding the substring threshold system.
#
# - AMBIGUITY_MARGIN: when the top retrieval lane (rag or memory)
#   beats the runner-up retrieval lane by less than this margin AND
#   the runner-up itself clears MIN_CONFIDENCE_FLOOR, the route is
#   upgraded from single-lane to "hybrid" so both retrievers run.
#
# Both gates are activated only when at least one Tier-2 embedding
# centroid is installed — see ``any_embedding_centroid`` in route().
# This keeps the router behaviorally identical to Tier 1 on installs
# that have not yet built the embedding centroids.
# ============================================================
MIN_CONFIDENCE_FLOOR = 0.30
AMBIGUITY_MARGIN     = 0.10


class CognitiveRouterV4:
    """
    Adaptive Cognitive Router (v4)

    ✔ Learns from recent routing history
    ✔ Adjusts retrieval aggressiveness dynamically
    ✔ Detects intent drift (conversation shift)
    ✔ Reduces RAG when latency increases
    ✔ Prevents retrieval overuse collapse
    ✔ NEW: Detects intent to search the Internet
    """

    def __init__(self):
        # -----------------------------
        # Adaptive State
        # -----------------------------
        self.latency_window = deque(maxlen=20)
        self.route_history = deque(maxlen=50)

        # Base thresholds for the normalized (in [0, 1]) substring
        # intent scorers. The scorers return ``hits / len(triggers)``,
        # so e.g. 2/7 ≈ 0.286 clears the 0.25 memory/rag base while
        # 1/7 ≈ 0.143 does not. The conservative values here are
        # paired with the pure threshold-recompute path in
        # ``_compute_dynamic_thresholds`` so latency adaptation and
        # tuner sensitivities can move the effective threshold in
        # both directions without compounding across turns.
        self.base_rag_threshold = 0.25
        self.base_memory_threshold = 0.25
        self.base_internet_threshold = 0.20

        self.dynamic_rag_threshold = self.base_rag_threshold
        self.dynamic_memory_threshold = self.base_memory_threshold
        self.dynamic_internet_threshold = self.base_internet_threshold  # NEW

        # Intent tracking
        self.last_intent_vector = None
        self.intent_drift_threshold = 0.35

        # Load control
        self.recent_rag_usage = deque(maxlen=10)

        # Phase B: semantic RECALL centroid + thresholds.
        # The centroid is set externally (typically by llm_worker on first
        # use, using ``workers.intent_router.build_centroid`` over a curated
        # example set). When unset we fall back to the substring detector
        # in core.memory_filters.detect_recall_intent.
        self.recall_centroid: np.ndarray | None = None
        # T4.2: semantic CHAT (negative class) centroid. Installed
        # symmetrically to ``recall_centroid`` by ``llm_worker``. When
        # unset, ``_score_chat_intent`` returns 0.0 and the margin gate
        # in ``route(...)`` is trivially satisfied — this preserves
        # backwards compatibility with fresh installs that haven't
        # installed the chat centroid yet.
        self.chat_centroid: np.ndarray | None = None
        # T4.2: raised from 0.55 to 0.62. The old single-centroid
        # classifier at 0.55 was too permissive — "Why is the sky blue?"
        # landed at ~0.61 against the recall centroid simply because it
        # shares high cosine geometry with "tell me about X" phrasings.
        self.recall_threshold = 0.62
        # T4.2: now USED. Recall must beat chat by this margin before
        # ``recall_active`` fires (see ``route(...)`` decision block).
        self.recall_margin_over_chat = 0.05

        # Tier 2: per-lane embedding centroids (memory / rag / web).
        # Each is L2-normalized and built externally by LLMWorker via
        # ``workers.intent_router.build_centroid``. When unset, the
        # corresponding ``_score_*_intent_embedding`` returns 0.0 so
        # the behavior reduces to Tier-1 substring-only — preserving
        # back-compat with installs that have not yet built the
        # centroids. The Tier-2 confidence layer in ``route(...)`` is
        # activated only when at least one of these is installed.
        self.memory_centroid: np.ndarray | None = None
        self.rag_centroid:    np.ndarray | None = None
        self.web_centroid:    np.ndarray | None = None

    # ============================================================
    # PUBLIC ENTRY
    # ============================================================

    def route(self, query: str, intent_vector=None, estimated_complexity=0.5, weights=None):
        """
        Returns routing decision dictionary
        """

        q = query.lower()

        # -----------------------------------------
        # 1. INTENT DRIFT DETECTION
        # -----------------------------------------
        drift = self._detect_intent_drift(intent_vector)

        # -----------------------------------------
        # 2. THRESHOLD COMPUTATION (pure, per-turn)
        #
        # Re-derive dynamic_*_threshold from base_*_threshold every
        # turn so that latency-adaptation and tuner-sensitivity
        # adjustments do not compound across calls. The fields are
        # still surfaced on ``self`` and in the decision dict for
        # telemetry/debug, but they are derived state, not
        # accumulating state.
        # -----------------------------------------
        sensitivities = weights or {}
        rag_thr, mem_thr, web_thr = self._compute_dynamic_thresholds(sensitivities)
        self.dynamic_rag_threshold = rag_thr
        self.dynamic_memory_threshold = mem_thr
        self.dynamic_internet_threshold = web_thr

        # -----------------------------------------
        # 3. LOAD-BASED RETRIEVAL SUPPRESSION
        # -----------------------------------------
        rag_penalty = self._rag_load_penalty()

        # -----------------------------------------
        # 4. DECISION SCORING
        # -----------------------------------------
        memory_score = self._score_memory_intent(q)
        rag_score = self._score_rag_intent(q)
        
        # 🔑 FIX: Changed to match the actual function name _score_web_intent
        web_score = self._score_web_intent(q)
        recall_score = self._score_recall_intent(intent_vector, q)
        chat_score = self._score_chat_intent(intent_vector, q)
        complexity_score = estimated_complexity

        # ----------------------------------------------------------
        # Tier 2: per-lane embedding scoring + fusion (max).
        #
        # Cosine score against per-lane centroids, reusing the SAME
        # ``intent_vector`` that drove recall/chat scoring (no extra
        # model call). When a centroid is unset the helper returns
        # 0.0, so on fresh installs the fusion ``max(substring,
        # embedding) == substring`` and behavior is identical to
        # Tier 1 — see ``any_embedding_centroid`` below for the gate
        # that also keeps the new confidence layer dormant.
        #
        # We deliberately use ``max`` (NOT an average): substring is a
        # high-precision boolean-ish signal at low recall; embeddings
        # add semantic recall without lowering substring precision.
        # ----------------------------------------------------------
        memory_score_embedding = self._score_memory_intent_embedding(intent_vector)
        rag_score_embedding    = self._score_rag_intent_embedding(intent_vector)
        web_score_embedding    = self._score_web_intent_embedding(intent_vector)

        memory_score_final = max(memory_score, memory_score_embedding)
        rag_score_final    = max(rag_score,    rag_score_embedding)
        web_score_final    = max(web_score,    web_score_embedding)

        # Per-lane source telemetry: which signal won the max(). Pure
        # observability — debugging behavior shifts after Tier 2 is
        # vastly easier when we can grep ``*_score_source`` in the
        # decision log instead of reverse-engineering it from raw
        # scores.
        memory_score_source = "embedding" if memory_score_embedding > memory_score else "substring"
        rag_score_source    = "embedding" if rag_score_embedding    > rag_score    else "substring"
        web_score_source    = "embedding" if web_score_embedding    > web_score    else "substring"

        # Apply dynamic thresholds against the FUSED score so semantic
        # recall can light up a lane that the substring triggers miss.
        memory_enabled = memory_score_final > self.dynamic_memory_threshold
        rag_enabled = rag_score_final > (self.dynamic_rag_threshold + rag_penalty)
        internet_enabled = web_score_final > self.dynamic_internet_threshold
        
        # Drift reduces retrieval sensitivity
        if drift:
            memory_enabled = False
            rag_enabled = False
            internet_enabled = False  # NEW

        # Phase B: semantic recall override forces HYBRID so memory + RAG
        # are fused for "tell me about X" / "who is X" queries. Web intent
        # still wins because the user explicitly asked to look online.
        #
        # T4.2: two gates — absolute threshold AND margin over the chat
        # (negative) class. When ``chat_centroid`` is unset,
        # ``chat_score`` is 0.0 so the margin requirement is trivially
        # satisfied, preserving backwards compatibility on fresh installs
        # that haven't installed the chat centroid yet.
        recall_active = (
            recall_score >= self.recall_threshold
            and (recall_score - chat_score) >= self.recall_margin_over_chat
        )

        # T4.2 observability: when the absolute threshold would have
        # fired but the margin over the chat class blocked it, surface
        # one INFO line so a human can grep ``logs/llm_debug.log`` for
        # the fix working. Silent config changes are the enemy.
        if recall_score >= self.recall_threshold and not recall_active:
            logger.info(
                "[RouterV4] recall blocked by chat-class margin "
                "(recall=%.3f, chat=%.3f, margin=%.3f, required=%.3f)",
                recall_score,
                chat_score,
                recall_score - chat_score,
                self.recall_margin_over_chat,
            )

        # High complexity forces hybrid
        if complexity_score > 0.75:
            route = "hybrid"
        elif internet_enabled:
            route = "web"  # NEW: internet-first route
        elif recall_active:
            route = "hybrid"
        elif rag_enabled and memory_enabled:
            route = "hybrid"
        elif rag_enabled:
            route = "rag"
        elif memory_enabled:
            route = "memory"
        else:
            route = "none"

        # ----------------------------------------------------------
        # Tier 2: confidence + ambiguity layer (additive, post-tree).
        #
        # Activation guard: only run when at least one Tier-2 centroid
        # is installed. This makes the layer a strict no-op on fresh
        # installs (centroids are built lazily by ``llm_worker``), so
        # Tier-2 cannot silently alter routing before the embedding
        # signals exist.
        # ----------------------------------------------------------
        any_embedding_centroid = (
            self.memory_centroid is not None
            or self.rag_centroid is not None
            or self.web_centroid is not None
        )

        # Identify the strongest intent across the four scored lanes.
        # ``recall_score`` belongs alongside the retrieval intents
        # because ``recall_active`` is what promotes a query to the
        # recall hybrid path.
        _intent_scores = {
            "memory": memory_score_final,
            "rag":    rag_score_final,
            "web":    web_score_final,
            "recall": recall_score,
        }
        top_intent = max(_intent_scores, key=_intent_scores.get)
        top_score = _intent_scores[top_intent]
        _ranked = sorted(_intent_scores.values(), reverse=True)
        second_best_score = _ranked[1] if len(_ranked) > 1 else 0.0
        confidence_margin = top_score - second_best_score

        # Identify which signal owned the top lane's final score.
        # Computed UNCONDITIONALLY (not gated by ``any_embedding_centroid``)
        # so the decision dict always carries ``top_intent_source`` —
        # invaluable for analyzing how often embeddings vs substrings
        # dominate when tuning ``MIN_CONFIDENCE_FLOOR`` later. Recall
        # is treated as ``substring`` here because the recall path is
        # gated separately by ``recall_threshold`` and the chat-margin
        # rule, not by the Tier-2 floor.
        top_intent_source = {
            "memory": memory_score_source,
            "rag":    rag_score_source,
            "web":    web_score_source,
            "recall": "substring",
        }[top_intent]

        if any_embedding_centroid:
            # ---- Confidence floor downgrade --------------------
            # The floor's stated purpose is to FILTER EMBEDDING
            # NOISE: cosine scores in [0.30, 0.40] are common even
            # for weak matches, so an embedding-only signal in that
            # band can be a false positive. Three gates therefore
            # apply, ALL of which must hold for a downgrade:
            #
            #   1. Embedding owned the top lane's score (substring
            #      is a high-precision keyword match — if it cleared
            #      the lane threshold, we trust it and bypass the
            #      floor entirely).
            #   2. The top score is below the absolute floor (0.30).
            #   3. The top score is also below the lane-relative
            #      buffer (lane_threshold + 0.05) — anchoring the
            #      floor to the existing threshold system so it
            #      acts as a SECONDARY guard, not a primary one.
            _lane_threshold = {
                "memory": self.dynamic_memory_threshold,
                "rag":    self.dynamic_rag_threshold,
                "web":    self.dynamic_internet_threshold,
                "recall": self.recall_threshold,
            }[top_intent]

            if (
                route in ("memory", "rag", "web", "hybrid")
                and not recall_active
                and top_intent_source == "embedding"
                and top_score < MIN_CONFIDENCE_FLOOR
                and top_score < _lane_threshold + 0.05
            ):
                logger.info(
                    "[RouterV4] confidence-floor downgrade route=%s -> none "
                    "(top_intent=%s, top_score=%.3f, lane_thr=%.3f, floor=%.2f)",
                    route, top_intent, top_score, _lane_threshold, MIN_CONFIDENCE_FLOOR,
                )
                route = "none"

            # ---- Ambiguity upgrade -----------------------------
            # When a single retrieval lane (rag or memory) won but the
            # runner-up retrieval lane is within AMBIGUITY_MARGIN AND
            # itself clears the floor, prefer hybrid so both retrievers
            # contribute. Web is excluded from the upgrade because an
            # internet route is qualitatively different from disk
            # retrieval. Recall-driven hybrid is already handled.
            elif (
                route in ("rag", "memory")
                and confidence_margin < AMBIGUITY_MARGIN
                and second_best_score >= MIN_CONFIDENCE_FLOOR
            ):
                # Only upgrade if the runner-up is the OTHER retrieval
                # lane (rag<->memory). If second-best is web/recall, a
                # rag/memory hybrid would not actually fuse the runner-up.
                _retrieval_pair = {"rag", "memory"}
                _ranked_intents = sorted(
                    _intent_scores.items(), key=lambda kv: kv[1], reverse=True
                )
                _second_intent = _ranked_intents[1][0] if len(_ranked_intents) > 1 else None
                if _second_intent in _retrieval_pair and _second_intent != top_intent:
                    logger.info(
                        "[RouterV4] ambiguity upgrade route=%s -> hybrid "
                        "(top=%s/%.3f, second=%s/%.3f, margin=%.3f<%.2f)",
                        route, top_intent, top_score,
                        _second_intent, second_best_score,
                        confidence_margin, AMBIGUITY_MARGIN,
                    )
                    route = "hybrid"

        decision = {
            "route": route,
            "drift": drift,
            "memory_score": memory_score,
            "rag_score": rag_score,
            "web_score": web_score, # 🔑 FIX: Tracked the corrected variable
            "recall_score": recall_score,
            "recall_active": recall_active,
            "chat_score": chat_score,
            "recall_margin_over_chat": self.recall_margin_over_chat,
            "rag_threshold": self.dynamic_rag_threshold,
            "memory_threshold": self.dynamic_memory_threshold,
            "internet_threshold": self.dynamic_internet_threshold,  # NEW
            "recall_threshold": self.recall_threshold,
            # Tier 2 additive observability fields. All keys are NEW;
            # nothing existing is renamed or removed so callers and
            # telemetry pipelines that only know the Tier-1 schema
            # continue to work unchanged.
            "memory_score_embedding": memory_score_embedding,
            "rag_score_embedding":    rag_score_embedding,
            "web_score_embedding":    web_score_embedding,
            "memory_score_final":     memory_score_final,
            "rag_score_final":        rag_score_final,
            "web_score_final":        web_score_final,
            "memory_score_source":    memory_score_source,
            "rag_score_source":       rag_score_source,
            "web_score_source":       web_score_source,
            "top_intent":             top_intent,
            "top_intent_source":      top_intent_source,
            "top_score":              top_score,
            "second_best_score":      second_best_score,
            "confidence_margin":      confidence_margin,
            "min_confidence_floor":   MIN_CONFIDENCE_FLOOR,
            "ambiguity_margin":       AMBIGUITY_MARGIN,
            "tier2_active":           any_embedding_centroid,
            "strategy": "adaptive_v4"
        }

        self.route_history.append((time.time(), route))
        logger.debug(f"[RouterV4] {decision}")

        return decision

    # ============================================================
    # INTENT SCORING (LIGHTWEIGHT SEMANTIC HEURISTIC)
    # ============================================================

    def _score_memory_intent(self, q: str) -> float:
        """Normalized substring score in [0, 1]: hits / len(triggers).

        Returning a float (instead of the prior raw integer count)
        makes the score type-comparable with the float thresholds in
        ``_compute_dynamic_thresholds`` so latency/tuner adaptivity
        can actually take effect on this lane.
        """
        triggers = _MEMORY_TRIGGERS
        if not triggers:
            return 0.0
        hits = sum(1 for t in triggers if t in q)
        return hits / len(triggers)

    def _score_rag_intent(self, q: str) -> float:
        """Normalized substring score in [0, 1]: hits / len(triggers).

        See ``_score_memory_intent`` for rationale.
        """
        triggers = _RAG_TRIGGERS
        if not triggers:
            return 0.0
        hits = sum(1 for t in triggers if t in q)
        return hits / len(triggers)
    
    def set_recall_centroid(self, centroid: np.ndarray) -> None:
        """Install the semantic centroid used by ``_score_recall_intent``.

        Built externally with ``workers.intent_router.build_centroid`` from a
        curated example set ("tell me about X", "who is X", "remind me
        about X", etc.). Called once on first use by ``llm_worker``.
        """
        try:
            self.recall_centroid = np.asarray(centroid, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[RouterV4] failed to install recall centroid: {e}")
            self.recall_centroid = None

    def set_chat_centroid(self, centroid: np.ndarray) -> None:
        """T4.2: install the semantic centroid for the general-knowledge /
        chat negative class.

        Built externally with ``workers.intent_router.build_centroid``
        from a curated example set (factual questions, coding snippets,
        short translations, etc. — see ``LLMWorker._CHAT_INTENT_EXAMPLES``).
        Called once on first use by ``llm_worker`` alongside
        ``set_recall_centroid``.
        """
        try:
            self.chat_centroid = np.asarray(centroid, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[RouterV4] failed to install chat centroid: {e}")
            self.chat_centroid = None

    def set_memory_centroid(self, centroid: np.ndarray) -> None:
        """Tier 2: install the semantic centroid for MEMORY intent.

        Built externally with ``workers.intent_router.build_centroid``
        from ``LLMWorker._MEMORY_INTENT_EXAMPLES``. Installed once on
        first use; the ``is None`` guard in ``_ensure_router_centroids``
        prevents stomping manual overrides on every turn.
        """
        try:
            self.memory_centroid = np.asarray(centroid, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[RouterV4] failed to install memory centroid: {e}")
            self.memory_centroid = None

    def set_rag_centroid(self, centroid: np.ndarray) -> None:
        """Tier 2: install the semantic centroid for RAG intent.

        Built externally with ``workers.intent_router.build_centroid``
        from ``LLMWorker._RAG_INTENT_EXAMPLES``. Symmetric to
        ``set_memory_centroid`` and ``set_web_centroid``.
        """
        try:
            self.rag_centroid = np.asarray(centroid, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[RouterV4] failed to install rag centroid: {e}")
            self.rag_centroid = None

    def set_web_centroid(self, centroid: np.ndarray) -> None:
        """Tier 2: install the semantic centroid for WEB intent.

        Built externally with ``workers.intent_router.build_centroid``
        from ``LLMWorker._WEB_INTENT_EXAMPLES``. Symmetric to
        ``set_memory_centroid`` and ``set_rag_centroid``.
        """
        try:
            self.web_centroid = np.asarray(centroid, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[RouterV4] failed to install web centroid: {e}")
            self.web_centroid = None

    def _cosine_against(self, intent_vector, centroid: np.ndarray | None) -> float:
        """Tier 2: shared cosine-similarity helper for the per-lane
        embedding scorers.

        Returns 0.0 when either ``intent_vector`` or ``centroid`` is
        unavailable so the embedding contribution to ``final_score =
        max(substring, embedding)`` is a safe no-op on fresh installs.
        """
        if intent_vector is None or centroid is None:
            return 0.0
        try:
            v = np.asarray(intent_vector, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn > 0:
                v = v / vn
            cn = np.linalg.norm(centroid)
            c = centroid / cn if cn > 0 else centroid
            return float(np.dot(v, c))
        except Exception as e:
            logger.debug(f"[RouterV4] embedding intent score failed: {e}")
            return 0.0

    def _score_memory_intent_embedding(self, intent_vector) -> float:
        """Tier 2: cosine score against the MEMORY centroid in [-1, 1].

        Returns 0.0 if the centroid was never installed or
        ``intent_vector`` is None — preserving back-compat.
        """
        return self._cosine_against(intent_vector, self.memory_centroid)

    def _score_rag_intent_embedding(self, intent_vector) -> float:
        """Tier 2: cosine score against the RAG centroid in [-1, 1].

        See ``_score_memory_intent_embedding`` for fallback behavior.
        """
        return self._cosine_against(intent_vector, self.rag_centroid)

    def _score_web_intent_embedding(self, intent_vector) -> float:
        """Tier 2: cosine score against the WEB centroid in [-1, 1].

        See ``_score_memory_intent_embedding`` for fallback behavior.
        """
        return self._cosine_against(intent_vector, self.web_centroid)

    def _score_chat_intent(self, intent_vector, q: str) -> float:
        """T4.2: symmetric to ``_score_recall_intent`` but against the
        chat (negative-class) centroid.

        Returns cosine similarity in [-1, 1] when a centroid + query
        vector are available; returns ``0.0`` otherwise (so the margin
        gate in ``route(...)`` stays trivially satisfied on installs
        where the chat centroid hasn't been built yet). The raw query
        string ``q`` is accepted for signature parity with
        ``_score_recall_intent`` but is intentionally unused — there is
        no substring fallback for the negative class.
        """
        del q  # unused; signature parity with _score_recall_intent
        if intent_vector is None or self.chat_centroid is None:
            return 0.0
        try:
            v = np.asarray(intent_vector, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn > 0:
                v = v / vn
            cn = np.linalg.norm(self.chat_centroid)
            c = self.chat_centroid / cn if cn > 0 else self.chat_centroid
            return float(np.dot(v, c))
        except Exception as e:
            logger.debug(f"[RouterV4] chat centroid score failed: {e}")
            return 0.0

    def _score_recall_intent(self, intent_vector, q: str) -> float:
        """Phase B semantic recall score in [0, 1].

        When an embedding centroid + query vector are available we use
        cosine similarity (Nomic v1.5 vectors are L2 normalized so dot
        product = cosine). Falls back to the substring detector from
        ``core.memory_filters.detect_recall_intent`` (returns 1.0 / 0.0)
        when no embedding is available.
        """
        if intent_vector is not None and self.recall_centroid is not None:
            try:
                v = np.asarray(intent_vector, dtype=np.float32)
                vn = np.linalg.norm(v)
                if vn > 0:
                    v = v / vn
                cn = np.linalg.norm(self.recall_centroid)
                c = self.recall_centroid / cn if cn > 0 else self.recall_centroid
                return float(np.dot(v, c))
            except Exception as e:
                logger.debug(f"[RouterV4] recall centroid score failed: {e}")

        # Substring fallback: 1.0 when a recall phrase is present, else 0.
        return 1.0 if detect_recall_intent(q) else 0.0

    def _score_web_intent(self, q: str) -> float:
        """Normalized substring score in [0, 1] for explicit
        intent to search the web. See ``_score_memory_intent``
        for rationale.
        """
        triggers = _WEB_TRIGGERS
        if not triggers:
            return 0.0
        hits = sum(1 for t in triggers if t in q)
        return hits / len(triggers)

    # ============================================================
    # ADAPTIVE BEHAVIOR
    # ============================================================

    def _latency_threshold_offset(self) -> float:
        """Pure read of ``self.latency_window`` returning a small
        additive offset to apply to every base threshold this turn.

        Returns +0.05 when the rolling average latency is high
        (tighten retrieval), -0.03 when it is low (loosen retrieval),
        and 0.0 otherwise (or when there are < 5 samples).
        """
        if len(self.latency_window) < 5:
            return 0.0
        avg_latency = float(np.mean(self.latency_window))
        if avg_latency > 2.5:
            return 0.05
        if avg_latency < 1.2:
            return -0.03
        return 0.0

    def _compute_dynamic_thresholds(self, sensitivities: dict) -> tuple[float, float, float]:
        """Pure function of (base thresholds, rolling latency, sensitivity weights).

        Returns ``(rag_threshold, memory_threshold, internet_threshold)``
        derived from each base via a sensitivity multiplier
        ``(2.0 - s)`` clamped to ``[0.5, 1.6]`` and an additive
        latency offset. Higher sensitivity (>1.0) lowers the
        threshold (more eager retrieval); lower sensitivity (<1.0)
        raises it.

        Clamp lower bounds are intentionally below the base values so
        a high sensitivity can pull the effective threshold below the
        base, preserving adaptive headroom in both directions.
        """
        lat_offset = self._latency_threshold_offset()

        def _derive(base: float, key: str, lo: float, hi: float) -> float:
            s = float(sensitivities.get(key, 1.0))
            mult = max(0.5, min(1.6, 2.0 - s))
            return max(lo, min(hi, base * mult + lat_offset))

        return (
            _derive(self.base_rag_threshold,      "rag_sensitivity",      0.10, 0.85),
            _derive(self.base_memory_threshold,   "memory_sensitivity",   0.10, 0.75),
            _derive(self.base_internet_threshold, "internet_sensitivity", 0.10, 0.90),
        )

    # ============================================================
    # DRIFT DETECTION
    # ============================================================

    def _detect_intent_drift(self, intent_vector):
        if self.last_intent_vector is None or intent_vector is None:
            self.last_intent_vector = intent_vector
            return False

        try:
            similarity = float(
                np.dot(self.last_intent_vector, intent_vector)
                / (np.linalg.norm(self.last_intent_vector) * np.linalg.norm(intent_vector) + 1e-8)
            )
        except Exception:
            similarity = 0.0

        self.last_intent_vector = intent_vector
        return similarity < self.intent_drift_threshold

    # ============================================================
    # LOAD CONTROL
    # ============================================================

    def _rag_load_penalty(self):
        if len(self.recent_rag_usage) == 0:
            return 0.0

        usage_ratio = sum(self.recent_rag_usage) / len(self.recent_rag_usage)

        # If RAG overused → increase threshold
        if usage_ratio > 0.6:
            return 0.1
        elif usage_ratio < 0.3:
            return -0.05

        return 0.0

    # ============================================================
    # FEEDBACK HOOKS (IMPORTANT)
    # ============================================================

    def record_latency(self, seconds: float):
        self.latency_window.append(seconds)

    def record_rag_used(self, used: bool):
        self.recent_rag_usage.append(1 if used else 0)