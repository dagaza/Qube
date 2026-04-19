import logging
from collections import deque

logger = logging.getLogger("Qube.RouterTuner")


class AdaptiveRouterSelfTunerV2:
    """
    Online router optimization layer.

    Learns from telemetry and adjusts routing sensitivity dynamically.
    """

    def __init__(self):
        self.history = deque(maxlen=200)

        # --- Dynamic thresholds ---
        self.hybrid_sensitivity = 1.0
        self.memory_sensitivity = 1.0
        self.rag_sensitivity = 1.0
        # Rides at 1.0 by default. No _adjust rule consumes it yet —
        # the field exists so CognitiveRouterV4 has a real source for
        # its ``weights.get("internet_sensitivity", 1.0)`` read instead
        # of always falling back to the default.
        self.internet_sensitivity = 1.0

        # --- Guardrails ---
        self.min_sensitivity = 0.4
        self.max_sensitivity = 2.0

    # ============================================================
    # INPUT: telemetry event
    # ============================================================
    def observe(self, event: dict):
        self.history.append(event)

        self._adjust(event)

    # ============================================================
    # CORE LEARNING LOOP
    # ============================================================
    def _adjust(self, event: dict):

        route = event.get("route")
        memory_hits = event.get("memory_hits", 0)
        rag_hits = event.get("rag_hits", 0)
        latency = event.get("latency_ms", 0)

        # ------------------------------------------------------------
        # 1. HYBRID OVERUSE DETECTION
        # ------------------------------------------------------------
        if route == "HYBRID":

            # expensive but useless
            if rag_hits == 0 and memory_hits == 0:
                self.hybrid_sensitivity *= 0.92
                logger.debug("[TUNER] Hybrid penalty applied (no retrieval hits)")

            # slow hybrid
            if latency > 1200:
                self.hybrid_sensitivity *= 0.90
                logger.debug("[TUNER] Hybrid penalized (latency spike)")

        # ------------------------------------------------------------
        # 2. MEMORY FAILURE DETECTION
        # ------------------------------------------------------------
        if route == "MEMORY" and memory_hits == 0:
            self.memory_sensitivity *= 0.95
            logger.debug("[TUNER] Memory recall weak → lowering threshold")

        # ------------------------------------------------------------
        # 3. RAG OVERUSE DETECTION
        # ------------------------------------------------------------
        if route in ["RAG", "HYBRID"]:

            if rag_hits > 6:
                self.rag_sensitivity *= 0.93
                logger.debug("[TUNER] RAG over-fetch → tightening retrieval")

            if latency > 1500:
                self.rag_sensitivity *= 0.90
                logger.debug("[TUNER] RAG latency spike → reducing usage")

        # ------------------------------------------------------------
        # clamp values
        # ------------------------------------------------------------
        self.hybrid_sensitivity = self._clamp(self.hybrid_sensitivity)
        self.memory_sensitivity = self._clamp(self.memory_sensitivity)
        self.rag_sensitivity = self._clamp(self.rag_sensitivity)
        self.internet_sensitivity = self._clamp(self.internet_sensitivity)

    def _clamp(self, v):
        return max(self.min_sensitivity, min(self.max_sensitivity, v))

    # ============================================================
    # EXTERNAL API (used by router)
    # ============================================================
    def get_weights(self):
        # Keys MUST match the names CognitiveRouterV4.route() reads
        # via ``weights.get("rag_sensitivity" / "memory_sensitivity"
        # / "internet_sensitivity", 1.0)``. Renaming on either side
        # is enforced by tests/test_router_tuner_router_contract.py.
        # ``hybrid_sensitivity`` is informational (the router does
        # not consume it; HYBRID is derived from rag_enabled and
        # memory_enabled).
        return {
            "rag_sensitivity":      self.rag_sensitivity,
            "memory_sensitivity":   self.memory_sensitivity,
            "internet_sensitivity": self.internet_sensitivity,
            "hybrid_sensitivity":   self.hybrid_sensitivity,
        }

    def debug(self):
        return {
            "hybrid_sensitivity": self.hybrid_sensitivity,
            "memory_sensitivity": self.memory_sensitivity,
            "rag_sensitivity": self.rag_sensitivity,
            "internet_sensitivity": self.internet_sensitivity,
            "samples": len(self.history),
        }