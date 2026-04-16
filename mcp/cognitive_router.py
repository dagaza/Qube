import time
import logging
from collections import deque, defaultdict
import numpy as np

from core.memory_filters import detect_recall_intent

logger = logging.getLogger("Qube.CognitiveRouterV4")


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

        self.base_rag_threshold = 0.55
        self.base_memory_threshold = 0.45
        self.base_internet_threshold = 0.6  # NEW: baseline for internet search

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
        self.recall_threshold = 0.55
        self.recall_margin_over_chat = 0.05

    # ============================================================
    # PUBLIC ENTRY
    # ============================================================

    def route(self, query: str, intent_vector=None, estimated_complexity=0.5, weights=None):
        """
        Returns routing decision dictionary
        """

        q = query.lower()

        # -----------------------------------------
        # 1. SYSTEM LATENCY ADAPTATION
        # -----------------------------------------
        self._update_latency_model()

        # -----------------------------------------
        # 2. INTENT DRIFT DETECTION
        # -----------------------------------------
        drift = self._detect_intent_drift(intent_vector)

        # -----------------------------------------
        # 3. APPLY WEIGHTS (OPTIONAL)
        # -----------------------------------------
        if weights:
            # Example: Lower threshold = higher sensitivity
            rag_sensitivity = weights.get("rag_sensitivity", 1.0)
            memory_sensitivity = weights.get("memory_sensitivity", 1.0)
            internet_sensitivity = weights.get("internet_sensitivity", 1.0)  # NEW

            # Apply the weights to the dynamic thresholds
            self.dynamic_rag_threshold *= (2.0 - rag_sensitivity) # Inverts so >1.0 lowers the threshold
            self.dynamic_memory_threshold *= (2.0 - memory_sensitivity)
            self.dynamic_internet_threshold *= (2.0 - internet_sensitivity)  # NEW

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
        complexity_score = estimated_complexity

        # Apply dynamic thresholds
        memory_enabled = memory_score > self.dynamic_memory_threshold
        rag_enabled = rag_score > (self.dynamic_rag_threshold + rag_penalty)
        
        # 🔑 FIX: Using the corrected web_score variable
        internet_enabled = web_score > self.dynamic_internet_threshold  
        
        # Drift reduces retrieval sensitivity
        if drift:
            memory_enabled = False
            rag_enabled = False
            internet_enabled = False  # NEW

        # Phase B: semantic recall override forces HYBRID so memory + RAG
        # are fused for "tell me about X" / "who is X" queries. Web intent
        # still wins because the user explicitly asked to look online.
        recall_active = recall_score >= self.recall_threshold

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

        decision = {
            "route": route,
            "drift": drift,
            "memory_score": memory_score,
            "rag_score": rag_score,
            "web_score": web_score, # 🔑 FIX: Tracked the corrected variable
            "recall_score": recall_score,
            "recall_active": recall_active,
            "rag_threshold": self.dynamic_rag_threshold,
            "memory_threshold": self.dynamic_memory_threshold,
            "internet_threshold": self.dynamic_internet_threshold,  # NEW
            "recall_threshold": self.recall_threshold,
            "strategy": "adaptive_v4"
        }

        self.route_history.append((time.time(), route))
        logger.debug(f"[RouterV4] {decision}")

        return decision

    # ============================================================
    # INTENT SCORING (LIGHTWEIGHT SEMANTIC HEURISTIC)
    # ============================================================

    def _score_memory_intent(self, q: str):
        triggers = [
            "remember", "did i", "do i like", "my preference",
            "what am i", "about me", "my settings"
        ]
        return sum(t in q for t in triggers)

    def _score_rag_intent(self, q: str):
        triggers = [
            "pdf", "document", "file", "notes",
            "according to", "in my", "library"
        ]
        return sum(t in q for t in triggers)
    
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

    def _score_web_intent(self, q: str):
        """NEW: detect clear user intent to search online"""
        triggers = [
            "look online", "search the web", "find on the internet",
            "google", "check online", "web search", "news", "current", 
            "latest", "today", "right now", "happening", "weather"
        ]
        return sum(t in q for t in triggers)

    # ============================================================
    # ADAPTIVE BEHAVIOR
    # ============================================================

    def _update_latency_model(self):
        if len(self.latency_window) < 5:
            return

        avg_latency = np.mean(self.latency_window)

        # If latency is high → reduce retrieval sensitivity
        if avg_latency > 2.5:
            self.dynamic_rag_threshold = min(0.7, self.dynamic_rag_threshold + 0.05)
            self.dynamic_memory_threshold = min(0.6, self.dynamic_memory_threshold + 0.05)
            self.dynamic_internet_threshold = min(0.75, self.dynamic_internet_threshold + 0.05)  # NEW
        # If system is fast → allow more retrieval
        elif avg_latency < 1.2:
            self.dynamic_rag_threshold = max(0.4, self.dynamic_rag_threshold - 0.03)
            self.dynamic_memory_threshold = max(0.3, self.dynamic_memory_threshold - 0.03)
            self.dynamic_internet_threshold = max(0.5, self.dynamic_internet_threshold - 0.03)  # NEW

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