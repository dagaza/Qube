import time
import logging
from collections import deque, defaultdict
import numpy as np

from core.memory_filters import detect_recall_intent
from mcp.router_lane_stats import (
    LANE_BIAS_DAMPING,
    LANE_BIAS_RAW_CLAMP,
    LaneStatsRegistry,
    RouteFeedbackEvent,
)
from mcp.routing_stability_tracker import (
    RoutingStabilityTracker,
    _DORMANT_DIAGNOSTIC,
)
from mcp.routing_policy_engine import (
    POLICY_ACCEPT,
    POLICY_NO_ACTION,
    RoutingPolicyDecision,
    RoutingPolicyEngine,
    RoutingPolicyInputs,
)
from mcp.routing_arbitration_layer import (
    INTERPRETATION_PASSTHROUGH,
    RoutingArbitrationDecision,
    RoutingArbitrationInputs,
    RoutingArbitrationLayer,
)

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

# Tier 3 decision-boundary band: damped lane bias from
# ``LaneStatsRegistry.adaptive_offset`` is applied to the per-lane
# dynamic thresholds ONLY when the post-fusion ``top_score`` lies in
# ``[MIN_CONFIDENCE_FLOOR, HIGH_CONFIDENCE_CEILING]``. Outside this
# band Tier 3 contributes exactly zero, so:
#   - clearly weak queries (top_score below the floor) are not
#     rescued by reliability-driven nudges,
#   - clearly strong embedding hits (top_score above the ceiling)
#     are never overpowered by Tier 3 — preserving the "embedding
#     dominance when strong" invariant.
HIGH_CONFIDENCE_CEILING = 0.75


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

        # Tier 3: bounded, observable feedback-driven calibration.
        # The registry holds per-lane reliability stats and exposes a
        # pure-read ``adaptive_offset(lane)`` consumed by
        # ``_compute_dynamic_thresholds``. Until any lane accumulates
        # ``MIN_OBSERVATIONS`` feedbacks the offset is identically
        # 0.0 — Tier 1 + Tier 2 behavior is bit-identical to today.
        # Feedback is fed in via ``observe_feedback(event)`` from
        # ``LLMWorker`` once per turn (after the post-retrieval
        # downgrade), so this object is the single Tier 3 state
        # container on the router.
        self.lane_stats = LaneStatsRegistry()

        # Tier 4 (RSL, v1 — observe-only): a bounded, embedding-gated
        # cluster bookkeeper that observes routing decisions across
        # semantically similar queries and surfaces stability
        # diagnostics in the decision dict. It does NOT modify the
        # ``route`` field in v1; override semantics are deferred to
        # a follow-up tier. State is in-memory and ephemeral per
        # worker restart. See ``mcp/routing_stability_tracker.py``.
        self.stability_tracker = RoutingStabilityTracker()

        # Tier 5 (RCPL, v1 — observability-only): a pure, stateless
        # policy synthesis engine that consumes Tier 2/3/4 signals
        # already produced earlier in ``route()`` and emits a
        # ``tier5_policy`` recommendation. v1 NEVER modifies the
        # ``route`` field. Engine is total over all inputs and
        # degrades to ``no_action`` on any unexpected exception.
        # See ``mcp/routing_policy_engine.py``.
        self.policy_engine = RoutingPolicyEngine()

        # Tier 6 (RAL, v1 — observability-only): a pure, stateless
        # arbitration layer that consumes Tier 2/3/4/5 signals
        # already produced earlier in ``route()`` and emits a
        # ``tier6_conflict_flags`` list plus a ``tier6_interpretation``
        # enum. v1 NEVER modifies the ``route`` field. Layer is
        # total over all inputs and degrades to a passthrough
        # sentinel on any unexpected exception.
        # See ``mcp/routing_arbitration_layer.py``.
        self.arbitration_layer = RoutingArbitrationLayer()

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

        # ----------------------------------------------------------
        # Tier 2 / Tier 3 prerequisite: derive top-intent metrics
        # BEFORE the per-lane threshold gates so the Tier 3 band-gate
        # can decide whether to apply a damped lane bias to the
        # thresholds. Recall is included alongside the retrieval
        # intents because ``recall_active`` promotes a query to the
        # recall hybrid path.
        #
        # ``top_intent`` / ``top_score`` / ``confidence_margin`` /
        # ``top_intent_source`` are all consumed unchanged by the
        # existing Tier 2 confidence + ambiguity layer further below
        # — they're just computed earlier now.
        # ----------------------------------------------------------
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
        top_intent_source = {
            "memory": memory_score_source,
            "rag":    rag_score_source,
            "web":    web_score_source,
            "recall": "substring",
        }[top_intent]

        # ----------------------------------------------------------
        # Tier 3: damped, band-gated lane bias on the dynamic
        # thresholds. The ``adaptive_offset`` raw signal lives in
        # ``[-LANE_BIAS_RAW_CLAMP, +LANE_BIAS_RAW_CLAMP]``; here it
        # is multiplied by ``LANE_BIAS_DAMPING`` and applied ONLY
        # when ``top_score`` lies in the decision-boundary band.
        #
        # Effective max influence per lane is therefore:
        #   |applied bias| <= LANE_BIAS_RAW_CLAMP * LANE_BIAS_DAMPING
        # which matches the per-lane ``[lo, hi]`` clamp expectation.
        #
        # ``top_score`` here is the post-fusion score (max of substring
        # and embedding per lane); the Tier 2 confidence + ambiguity
        # layer below reads the SAME ``top_score`` value.
        # ----------------------------------------------------------
        tier3_band_active = (MIN_CONFIDENCE_FLOOR <= top_score <= HIGH_CONFIDENCE_CEILING)
        if tier3_band_active:
            memory_lane_bias = self.lane_stats.adaptive_offset("memory") * LANE_BIAS_DAMPING
            rag_lane_bias    = self.lane_stats.adaptive_offset("rag")    * LANE_BIAS_DAMPING
            web_lane_bias    = self.lane_stats.adaptive_offset("web")    * LANE_BIAS_DAMPING
            self.dynamic_memory_threshold   = self._clamp_lane_threshold(
                "memory", self.dynamic_memory_threshold + memory_lane_bias
            )
            self.dynamic_rag_threshold      = self._clamp_lane_threshold(
                "rag", self.dynamic_rag_threshold + rag_lane_bias
            )
            self.dynamic_internet_threshold = self._clamp_lane_threshold(
                "web", self.dynamic_internet_threshold + web_lane_bias
            )
        else:
            memory_lane_bias = 0.0
            rag_lane_bias    = 0.0
            web_lane_bias    = 0.0

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

        # NOTE: ``top_intent`` / ``top_score`` / ``second_best_score``
        # / ``confidence_margin`` / ``top_intent_source`` are computed
        # earlier (right after fusion) so the Tier 3 band-gate can
        # consume ``top_score`` BEFORE the per-lane gates. The Tier 2
        # confidence + ambiguity layer below reuses those same values
        # unchanged; the routing semantics are identical to the
        # previous arrangement.
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

        # ----------------------------------------------------------
        # Tier 4: routing stability tracker (observe-only, v1).
        #
        # Hooks in AFTER the Tier 2 confidence + ambiguity layer has
        # finalized ``route``, so the diagnostic reflects the actual
        # outgoing decision. v1 is strictly observability — the
        # ``route`` field is NEVER modified by this block. Override
        # semantics are deferred to a follow-up tier.
        #
        # Defense-in-depth: ``observe(...)`` is already total over
        # all inputs, but we wrap it in try/except so even an
        # unforeseen numpy edge case can't kill a user-facing turn.
        # ----------------------------------------------------------
        try:
            tier4_diag = self.stability_tracker.observe(intent_vector, route)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.warning("[RouterV4] tier4 observe raised: %r — degrading to dormant.", exc)
            tier4_diag = _DORMANT_DIAGNOSTIC
        if tier4_diag.is_oscillating:
            logger.info(
                "[RouterV4] tier4 oscillation cluster_id=%s size=%d dominant=%s freq=%.2f current=%s",
                tier4_diag.cluster_id, tier4_diag.cluster_size,
                tier4_diag.dominant_route, tier4_diag.dominant_frequency, route,
            )

        # ----------------------------------------------------------
        # Tier 5: routing policy engine (observability-only, v1).
        #
        # Pure, stateless synthesis layer over Tier 2/3/4 signals
        # already in scope. Emits a ``tier5_policy`` recommendation
        # but NEVER modifies the ``route`` field in v1. Defense-in-
        # depth: ``evaluate(...)`` is internally total over all
        # inputs, but we still wrap it in try/except so even a
        # misbehaving engine cannot kill a user-facing turn.
        # ----------------------------------------------------------
        try:
            _intent_scores_for_policy = {
                "rag":    rag_score_final,
                "memory": memory_score_final,
                "web":    web_score_final,
                "recall": recall_score,
                "chat":   chat_score,
            }
            _ranked_for_policy = sorted(
                _intent_scores_for_policy.items(),
                key=lambda kv: kv[1], reverse=True,
            )
            _second_intent_for_policy = (
                _ranked_for_policy[1][0] if len(_ranked_for_policy) > 1 else None
            )
            tier5_inputs = RoutingPolicyInputs(
                confidence_margin=confidence_margin,
                top_intent=top_intent,
                top_score=top_score,
                second_best_score=second_best_score,
                second_best_intent=_second_intent_for_policy,
                tier2_active=any_embedding_centroid,
                memory_lane_bias=memory_lane_bias,
                rag_lane_bias=rag_lane_bias,
                web_lane_bias=web_lane_bias,
                tier3_band_active=tier3_band_active,
                tier4_active=tier4_diag.active,
                tier4_cluster_size=tier4_diag.cluster_size,
                tier4_cluster_dominant_frequency=tier4_diag.dominant_frequency,
                tier4_cluster_oscillating=tier4_diag.is_oscillating,
            )
            tier5_decision = self.policy_engine.evaluate(tier5_inputs)
            tier5_active = True
        except Exception as exc:  # pragma: no cover - defensive only
            logger.warning(
                "[RouterV4] tier5 evaluate raised: %r — degrading to no_action.", exc,
            )
            tier5_decision = RoutingPolicyDecision(POLICY_NO_ACTION, None, {})
            tier5_active = False

        if (
            tier5_decision.policy != POLICY_ACCEPT
            and tier5_decision.policy != POLICY_NO_ACTION
        ):
            logger.info(
                "[Tier5Policy] policy=%s reason=%s confidence_margin=%.3f oscillation=%.3f",
                tier5_decision.policy, tier5_decision.reason,
                confidence_margin,
                tier5_decision.inputs_snapshot.get("oscillation_index", 0.0),
            )

        # ----------------------------------------------------------
        # 8. TIER 6 (RAL v1, observability-only) — arbitration
        # ----------------------------------------------------------
        # Pure, side-effect-free conflict-resolution layer. Reads
        # the Tier 5 output plus the same Tier 2/3/4 signals Tier 5
        # used and surfaces a list of cross-tier conflict flags +
        # an interpretation enum. NEVER touches the ``route`` field
        # in v1; pinned by ``Tier6RouteFieldNeverModified``.
        try:
            tier6_inputs = RoutingArbitrationInputs(
                confidence_margin=confidence_margin,
                top_intent=top_intent,
                memory_lane_bias=memory_lane_bias,
                rag_lane_bias=rag_lane_bias,
                web_lane_bias=web_lane_bias,
                tier4_active=tier4_diag.active,
                tier4_cluster_dominant_frequency=tier4_diag.dominant_frequency,
                tier4_cluster_oscillating=tier4_diag.is_oscillating,
                tier5_policy=tier5_decision.policy,
                tier5_policy_reason=tier5_decision.reason,
            )
            tier6_decision = self.arbitration_layer.evaluate(tier6_inputs)
            tier6_active = True
        except Exception as exc:  # pragma: no cover - defensive only
            logger.warning(
                "[RouterV4] tier6 evaluate raised: %r — degrading to passthrough.",
                exc,
            )
            tier6_decision = RoutingArbitrationDecision(
                (), INTERPRETATION_PASSTHROUGH, {},
            )
            tier6_active = False

        if tier6_decision.conflict_flags:
            logger.info(
                "[Tier6RAL] conflict=%s interpretation=%s stability=%.3f policy=%s",
                ",".join(tier6_decision.conflict_flags),
                tier6_decision.interpretation,
                tier6_decision.inputs_snapshot.get("stability_score", 0.0),
                tier5_decision.policy,
            )

        # ---- Decision trace (observability-only) ----
        # Pure read-only explanation of WHY this ``route`` was selected.
        # Classifies the priority-tree branch / Tier 2 post-tree modifier
        # that fired, lists losing candidates with reasons, and surfaces
        # the Tier 3/4/5/6 context. Zero impact on routing behavior —
        # pinned by the null-trace random-walk control test.
        trace = self._build_decision_trace(
            route=route,
            drift=drift,
            recall_active=recall_active,
            intent_vector=intent_vector,
            complexity_score=complexity_score,
            rag_penalty=rag_penalty,
            memory_score_final=memory_score_final,
            rag_score_final=rag_score_final,
            web_score_final=web_score_final,
            recall_score=recall_score,
            memory_score_source=memory_score_source,
            rag_score_source=rag_score_source,
            web_score_source=web_score_source,
            memory_enabled=memory_enabled,
            rag_enabled=rag_enabled,
            internet_enabled=internet_enabled,
            any_embedding_centroid=any_embedding_centroid,
            top_intent=top_intent,
            top_intent_source=top_intent_source,
            top_score=top_score,
            second_best_score=second_best_score,
            confidence_margin=confidence_margin,
            memory_lane_bias=memory_lane_bias,
            rag_lane_bias=rag_lane_bias,
            web_lane_bias=web_lane_bias,
            tier3_band_active=tier3_band_active,
            tier4_diag=tier4_diag,
            tier5_decision=tier5_decision,
            tier5_active=tier5_active,
            tier6_decision=tier6_decision,
            tier6_active=tier6_active,
        )
        logger.debug("[RouterV4Trace] %s", trace)

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
            # Tier 3 additive observability fields. The ``*_lane_bias``
            # values are the APPLIED bias for this turn (zero when
            # ``tier3_band_active`` is False, ``adaptive_offset(lane) *
            # LANE_BIAS_DAMPING`` when True). Effective max magnitude
            # is therefore ``LANE_BIAS_RAW_CLAMP * LANE_BIAS_DAMPING``
            # (== 0.03). The ``*_lane_success_rate`` /
            # ``*_embedding_trust`` / ``*_substring_trust`` keys are
            # observability-only diagnostics from ``LaneStatsRegistry``
            # and never feed back into routing decisions.
            "memory_lane_bias":              memory_lane_bias,
            "rag_lane_bias":                 rag_lane_bias,
            "web_lane_bias":                 web_lane_bias,
            "memory_lane_success_rate":      self.lane_stats.recent_success_rate("memory"),
            "rag_lane_success_rate":         self.lane_stats.recent_success_rate("rag"),
            "web_lane_success_rate":         self.lane_stats.recent_success_rate("web"),
            "memory_embedding_trust":        self.lane_stats.embedding_trust("memory"),
            "rag_embedding_trust":           self.lane_stats.embedding_trust("rag"),
            "web_embedding_trust":           self.lane_stats.embedding_trust("web"),
            "memory_substring_trust":        self.lane_stats.substring_trust("memory"),
            "rag_substring_trust":           self.lane_stats.substring_trust("rag"),
            "web_substring_trust":           self.lane_stats.substring_trust("web"),
            "tier3_active":                  self.lane_stats.tier3_active(),
            "tier3_band_active":             tier3_band_active,
            "tier3_high_confidence_ceiling": HIGH_CONFIDENCE_CEILING,
            "tier3_damping":                 LANE_BIAS_DAMPING,
            # Tier 4 additive observability fields (RSL v1, observe-only).
            # All keys are NEW; nothing existing is renamed or removed.
            # ``tier4_active`` is the canonical "did the tracker actually
            # run this turn?" signal — False when ``intent_vector`` is
            # absent or unusable. The remaining fields carry sentinel
            # values in that case.
            "tier4_active":                       tier4_diag.active,
            "tier4_cluster_id":                   tier4_diag.cluster_id,
            "tier4_cluster_size":                 tier4_diag.cluster_size,
            "tier4_cluster_dominant_route":       tier4_diag.dominant_route,
            "tier4_cluster_dominant_frequency":   tier4_diag.dominant_frequency,
            "tier4_cluster_oscillating":          tier4_diag.is_oscillating,
            "tier4_route_consistent_with_cluster": tier4_diag.route_consistent_with_cluster,
            "tier4_total_clusters":               len(self.stability_tracker.clusters),
            # Tier 5 additive observability fields (RCPL v1, observability-only).
            # All keys are NEW; nothing existing is renamed or removed.
            # ``tier5_active`` is False ONLY on the failure path
            # (engine raised); ``accept`` and any rule-fired policy
            # still report ``tier5_active=True``.
            "tier5_active":         tier5_active,
            "tier5_policy":         tier5_decision.policy,
            "tier5_policy_reason":  tier5_decision.reason,
            "tier5_policy_inputs":  tier5_decision.inputs_snapshot,
            # Tier 6 additive observability fields (RAL v1, observability-only).
            # All keys are NEW; nothing existing is renamed or removed.
            # ``tier6_active`` is False ONLY on the failure path (layer
            # raised); ``stable`` and any rule-fired interpretation
            # still report ``tier6_active=True``. ``tier6_conflict_flags``
            # is the (possibly empty) list of fired flags in
            # ``CONFLICT_FLAGS_VALUES`` order — empty list means "no
            # conflict". The route field is byte-identical with or
            # without this hook (pinned by Tier6RouteFieldNeverModified).
            "tier6_active":           tier6_active,
            "tier6_conflict_flags":   list(tier6_decision.conflict_flags),
            "tier6_interpretation":   tier6_decision.interpretation,
            "tier6_inputs_snapshot":  tier6_decision.inputs_snapshot,
            # Decision trace (observability-only). Pure read-only
            # explanation built from already-computed values; zero
            # impact on routing behavior. See ``_build_decision_trace``.
            "trace": trace,
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

    # Per-lane safe range for the dynamic threshold. Single source of
    # truth shared by ``_compute_dynamic_thresholds`` and the Tier 3
    # band-gate apply step in ``route()`` so the clamp values cannot
    # drift apart between the two call sites.
    _LANE_THRESHOLD_BOUNDS: dict = {
        "rag":    (0.10, 0.85),
        "memory": (0.10, 0.75),
        "web":    (0.10, 0.90),
    }

    def _clamp_lane_threshold(self, lane: str, thr: float) -> float:
        lo, hi = self._LANE_THRESHOLD_BOUNDS.get(lane, (0.0, 1.0))
        return max(lo, min(hi, thr))

    def _compute_dynamic_thresholds(self, sensitivities: dict) -> tuple[float, float, float]:
        """Pure function of (base thresholds, rolling latency, sensitivity weights).

        Returns ``(rag_threshold, memory_threshold, internet_threshold)``
        derived from each base via a sensitivity multiplier
        ``(2.0 - s)`` clamped to ``[0.5, 1.6]`` and an additive
        latency offset. Higher sensitivity (>1.0) lowers the
        threshold (more eager retrieval); lower sensitivity (<1.0)
        raises it.

        Tier 3 calibration is intentionally NOT folded into this
        function. The lane-bias contribution is added in ``route()``
        AFTER ``top_score`` is known, and ONLY when ``top_score`` sits
        inside ``[MIN_CONFIDENCE_FLOOR, HIGH_CONFIDENCE_CEILING]``.
        Keeping this function bias-free preserves the property that
        ``_compute_dynamic_thresholds`` is a pure deterministic
        derivation from base thresholds, sensitivity, and latency.

        Clamp lower bounds are intentionally below the base values so
        a high sensitivity can pull the effective threshold below the
        base, preserving adaptive headroom in both directions.
        """
        lat_offset = self._latency_threshold_offset()

        def _derive(base: float, key: str, lane: str) -> float:
            s = float(sensitivities.get(key, 1.0))
            mult = max(0.5, min(1.6, 2.0 - s))
            return self._clamp_lane_threshold(lane, base * mult + lat_offset)

        return (
            _derive(self.base_rag_threshold,      "rag_sensitivity",      "rag"),
            _derive(self.base_memory_threshold,   "memory_sensitivity",   "memory"),
            _derive(self.base_internet_threshold, "internet_sensitivity", "web"),
        )

    # ============================================================
    # TIER 3: feedback ingestion
    # ============================================================

    def observe_feedback(self, event: RouteFeedbackEvent) -> None:
        """Tier 3: ingest one ``RouteFeedbackEvent`` from ``LLMWorker``.

        Thin wrapper over ``self.lane_stats.update(event)`` so the
        worker only knows about one router-side API. Defensive
        try/except: a feedback failure must NEVER crash a user-facing
        turn.
        """
        try:
            self.lane_stats.update(event)
        except Exception as e:
            logger.warning(f"[RouterV4] observe_feedback failed: {e}")

    # ============================================================
    # DECISION TRACE (OBSERVABILITY-ONLY)
    # ============================================================
    #
    # A pure, read-only helper that classifies the routing decision
    # produced by Tier 1-6 into a structured explanation dict. It
    # never re-scores, never re-runs threshold logic, never mutates
    # any field on ``self`` or on the caller's locals. The returned
    # dict is splatted into ``decision["trace"]`` by ``route()``.
    #
    # Every numeric leaf is coerced to a plain Python ``float`` so
    # the trace is JSON-safe; ``None`` is reserved for known-absent
    # optionals (Tier 4 dormant cluster fields).

    def _build_decision_trace(
        self,
        *,
        route: str,
        drift: bool,
        recall_active: bool,
        intent_vector,
        complexity_score: float,
        rag_penalty: float,
        memory_score_final: float,
        rag_score_final: float,
        web_score_final: float,
        recall_score: float,
        memory_score_source: str,
        rag_score_source: str,
        web_score_source: str,
        memory_enabled: bool,
        rag_enabled: bool,
        internet_enabled: bool,
        any_embedding_centroid: bool,
        top_intent: str,
        top_intent_source: str,
        top_score: float,
        second_best_score: float,
        confidence_margin: float,
        memory_lane_bias: float,
        rag_lane_bias: float,
        web_lane_bias: float,
        tier3_band_active: bool,
        tier4_diag,
        tier5_decision,
        tier5_active: bool,
        tier6_decision,
        tier6_active: bool,
    ) -> dict:
        """Build ``decision["trace"]`` — observability-only.

        Classification over already-computed state. See the plan
        ``decision_trace_observability_layer`` for the winning_reason
        and losing_candidates taxonomies.
        """
        # ---- Lane thresholds (post-Tier-3 band-gate) ----
        # These are the exact numbers the gates used this turn:
        # dynamic_*_threshold is already post-band-gate, and rag_penalty
        # is added at the rag gate only.
        lane_thresholds: dict[str, float] = {
            "memory": float(self.dynamic_memory_threshold),
            "rag":    float(self.dynamic_rag_threshold + rag_penalty),
            "web":    float(self.dynamic_internet_threshold),
            "recall": float(self.recall_threshold),
        }
        lane_scores_final: dict[str, float] = {
            "memory": float(memory_score_final),
            "rag":    float(rag_score_final),
            "web":    float(web_score_final),
            "recall": float(recall_score),
        }
        lane_sources: dict[str, str] = {
            "memory": memory_score_source,
            "rag":    rag_score_source,
            "web":    web_score_source,
            "recall": "substring",  # matches existing top_intent_source convention
        }

        # ---- Tier 2 post-tree modifier inference (section 2.4) ----
        # Predicates over already-decided state. We read, never recompute.
        complexity_forced = bool(complexity_score > 0.75)

        # Floor fired iff the exact router predicate at lines ~456-462
        # would be True given the final ``route``.
        floor_applied = bool(
            route == "none"
            and any_embedding_centroid
            and not recall_active
            and top_intent_source == "embedding"
            and top_score < MIN_CONFIDENCE_FLOOR
            and top_score < lane_thresholds.get(top_intent, 0.0) + 0.05
        )

        # Ambiguity fired iff the router predicate at lines ~477-498
        # would be True AND the resulting route is ``hybrid`` AND no
        # earlier branch claimed the hybrid label.
        _retrieval_pair = {"rag", "memory"}
        _ranked_intents_pairs = sorted(
            lane_scores_final.items(), key=lambda kv: kv[1], reverse=True,
        )
        _second_intent_name = (
            _ranked_intents_pairs[1][0] if len(_ranked_intents_pairs) > 1 else None
        )
        ambiguity_applied = bool(
            route == "hybrid"
            and not complexity_forced
            and not recall_active
            and not (rag_enabled and memory_enabled)
            and any_embedding_centroid
            and confidence_margin < AMBIGUITY_MARGIN
            and second_best_score >= MIN_CONFIDENCE_FLOOR
            and top_intent in _retrieval_pair
            and _second_intent_name in _retrieval_pair
            and _second_intent_name != top_intent
        )

        # ---- winning_reason (section 2.1 + 2.2) ----
        # Classification of which branch of the priority tree (or
        # Tier 2 post-tree modifier) set the final ``route``.
        if floor_applied:
            winning_reason = "confidence_floor_downgrade_to_none"
        elif ambiguity_applied:
            winning_reason = "ambiguity_upgrade_to_hybrid"
        elif route == "hybrid":
            if complexity_forced:
                winning_reason = "complexity_forced_hybrid"
            elif recall_active:
                winning_reason = "recall_override_hybrid"
            elif rag_enabled and memory_enabled:
                winning_reason = "dual_threshold_hybrid"
            else:
                # Defensive: should be unreachable given the tree, but
                # a future edit to the tree should not crash the trace.
                winning_reason = "hybrid_unknown"
        elif route == "web":
            winning_reason = "internet_enabled"
        elif route == "rag":
            winning_reason = "single_rag"
        elif route == "memory":
            winning_reason = "single_memory"
        elif route == "none":
            winning_reason = "no_lane_cleared_threshold"
        else:
            # Defensive: unknown route label.
            winning_reason = "unknown_route"

        # ---- winning_signal ----
        # Maps the final route to a coherent (lane, score, threshold,
        # source) tuple. For hybrid-via-recall, report the recall lane
        # explicitly; otherwise, report the top_intent lane (it is the
        # strongest semantic signal regardless of which branch fired).
        if route == "hybrid" and recall_active and not (
            complexity_forced or (rag_enabled and memory_enabled) or ambiguity_applied
        ):
            winning_lane = "recall"
            winning_score = float(recall_score)
            winning_threshold = float(self.recall_threshold)
            winning_source = "substring"
        elif route in ("memory", "rag", "web"):
            winning_lane = route
            winning_score = lane_scores_final[route]
            winning_threshold = lane_thresholds[route]
            winning_source = lane_sources[route]
        elif route == "hybrid":
            # complexity_forced / dual_threshold / ambiguity_upgrade:
            # top_intent is the dominant lane signal.
            winning_lane = "hybrid"
            winning_score = float(top_score)
            winning_threshold = float(lane_thresholds.get(top_intent, 0.0))
            winning_source = top_intent_source
        else:
            # route == "none": report top_intent's numbers — it is
            # still the best lane, just below the bar.
            winning_lane = "none"
            winning_score = float(top_score)
            winning_threshold = float(lane_thresholds.get(top_intent, 0.0))
            winning_source = top_intent_source

        # ---- losing_candidates (section 2.3) ----
        # Stable order: memory, rag, web, recall. Winner is excluded.
        # In hybrid-fused cases (dual_threshold, ambiguity_upgrade),
        # both memory and rag get reason="hybrid_fused" so the caller
        # can see they are BOTH part of the decision.
        hybrid_fused_lanes: set[str] = set()
        if winning_reason in ("dual_threshold_hybrid", "ambiguity_upgrade_to_hybrid"):
            hybrid_fused_lanes = {"memory", "rag"}

        lane_enabled: dict[str, bool] = {
            "memory": bool(memory_enabled),
            "rag":    bool(rag_enabled),
            "web":    bool(internet_enabled),
            # Recall has its own two-gate predicate; "enabled" here
            # means ``recall_active``.
            "recall": bool(recall_active),
        }

        losing_candidates: list[dict] = []
        for lane in ("memory", "rag", "web", "recall"):
            # Skip the sole winner.
            if lane == winning_lane:
                continue

            lane_score = lane_scores_final[lane]
            lane_threshold = lane_thresholds[lane]

            # Recall has its own below-threshold vs blocked-by-margin
            # distinction.
            if lane == "recall":
                if recall_score >= self.recall_threshold and not recall_active:
                    reason = "blocked_by_chat_margin"
                else:
                    reason = "below_threshold"
            elif lane in hybrid_fused_lanes:
                reason = "hybrid_fused"
            elif floor_applied and lane == top_intent:
                reason = "confidence_floor_downgraded_winner"
            elif drift and lane in ("memory", "rag", "web"):
                reason = "drift_suppressed"
            elif not lane_enabled[lane]:
                reason = "below_threshold"
            else:
                # Lane cleared its threshold but another lane was picked.
                reason = "lower_score_than_winner"

            losing_candidates.append({
                "lane":      lane,
                "score":     float(lane_score),
                "threshold": float(lane_threshold),
                "reason":    reason,
            })

        # ---- Tier 4 block ----
        # tier4_diag is always a ClusterDiagnostic; its optional fields
        # are already ``None`` when the tracker is dormant. Coerce
        # numeric optionals to plain floats (leave ``None`` alone).
        _cluster_freq = tier4_diag.dominant_frequency
        tier4_block = {
            "active":                     bool(tier4_diag.active),
            "cluster_id":                 tier4_diag.cluster_id,
            "cluster_size":               int(tier4_diag.cluster_size),
            "cluster_dominant_route":     tier4_diag.dominant_route,
            "cluster_dominant_frequency": (
                float(_cluster_freq) if _cluster_freq is not None else None
            ),
            "cluster_oscillating":        bool(tier4_diag.is_oscillating),
        }

        # ---- Assemble ----
        return {
            "selected_route": str(route),
            "winning_reason": winning_reason,
            "winning_signal": {
                "lane":      winning_lane,
                "score":     float(winning_score),
                "threshold": float(winning_threshold),
                "source":    winning_source,
            },
            "losing_candidates": losing_candidates,
            "confidence": {
                "top_intent":        top_intent,
                "top_intent_source": top_intent_source,
                "top_score":         float(top_score),
                "second_best_score": float(second_best_score),
                "margin":            float(confidence_margin),
                "floor":             float(MIN_CONFIDENCE_FLOOR),
                "ambiguity_margin":  float(AMBIGUITY_MARGIN),
                "floor_applied":     floor_applied,
                "ambiguity_applied": ambiguity_applied,
                "tier2_active":      bool(any_embedding_centroid),
            },
            "tier3": {
                "band_active":             bool(tier3_band_active),
                "high_confidence_ceiling": float(HIGH_CONFIDENCE_CEILING),
                "damping":                 float(LANE_BIAS_DAMPING),
                "lane_bias": {
                    "memory": float(memory_lane_bias),
                    "rag":    float(rag_lane_bias),
                    "web":    float(web_lane_bias),
                },
            },
            "tier4": tier4_block,
            "tier5_6": {
                "tier5_active":   bool(tier5_active),
                "policy":         str(tier5_decision.policy),
                "policy_reason":  tier5_decision.reason,
                "tier6_active":   bool(tier6_active),
                "conflicts":      list(tier6_decision.conflict_flags),
                "interpretation": str(tier6_decision.interpretation),
            },
            "context": {
                "drift":                 bool(drift),
                "recall_active":         bool(recall_active),
                "complexity_forced":     complexity_forced,
                "rag_penalty":           float(rag_penalty),
                "intent_vector_present": intent_vector is not None,
            },
        }

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