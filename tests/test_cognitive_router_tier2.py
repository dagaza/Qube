"""Tier 2 tests for ``CognitiveRouterV4``:

* per-lane embedding scoring (memory / rag / web)
* fused ``final_score = max(substring, embedding)`` and per-lane
  ``*_score_source`` telemetry
* confidence layer (``MIN_CONFIDENCE_FLOOR`` floor + tightened
  lane-relative buffer; ``AMBIGUITY_MARGIN`` upgrade)
* the ``any_embedding_centroid`` activation guard
* decision-dict additive contract
* regression: explicit substring queries still route correctly

We follow the convention of ``tests/test_cognitive_router_margin.py``
— pure-numpy synthetic centroids on orthogonal axes so the cosine
between an intent vector and each centroid is just a coordinate
read. No Qt / audio / embedder imports.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.cognitive_router import (
    CognitiveRouterV4,
    MIN_CONFIDENCE_FLOOR,
    AMBIGUITY_MARGIN,
)


# ============================================================
# Synthetic centroids on orthogonal axes in R^5:
#   axis 0 -> memory
#   axis 1 -> rag
#   axis 2 -> web
#   axis 3 -> recall (kept dormant unless the test installs it)
#   axis 4 -> chat   (kept dormant unless the test installs it)
#
# Because the centroids are orthogonal, an intent vector with
# components ``(m, r, w, rc, ch)`` produces exactly those cosine
# scores against each centroid, provided ``sum(c**2) <= 1``.
# ============================================================
_MEM_AXIS    = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_RAG_AXIS    = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_WEB_AXIS    = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_RECALL_AXIS = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_CHAT_AXIS   = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _intent_vec(m: float = 0.0, r: float = 0.0, w: float = 0.0,
                rc: float = 0.0, ch: float = 0.0) -> np.ndarray:
    """Build a unit vector with the requested cosines on each axis.

    Pads the residual on a 6th component if needed; raises if the
    requested components together exceed unit length.
    """
    comps = [m, r, w, rc, ch]
    sq = sum(c * c for c in comps)
    if sq > 1.0:
        raise ValueError(
            f"Cosines exceed unit length: {comps} (sum of squares={sq:.4f})."
        )
    residual = math.sqrt(max(0.0, 1.0 - sq))
    v = np.array(comps + [residual], dtype=np.float32)
    # Pad centroid axes to the same dim with zeros at the residual
    # slot so the dot product is unchanged.
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v


def _pad6(centroid: np.ndarray) -> np.ndarray:
    """Pad a length-5 axis to length 6 (the residual axis used by the
    intent vector), zero-padded so the dot product is unchanged."""
    out = np.zeros(6, dtype=np.float32)
    out[: centroid.shape[0]] = centroid
    return out


def _install_tier2_centroids(router: CognitiveRouterV4) -> None:
    router.set_memory_centroid(_pad6(_MEM_AXIS))
    router.set_rag_centroid(_pad6(_RAG_AXIS))
    router.set_web_centroid(_pad6(_WEB_AXIS))


def _install_recall_chat_centroids(router: CognitiveRouterV4) -> None:
    router.set_recall_centroid(_pad6(_RECALL_AXIS))
    router.set_chat_centroid(_pad6(_CHAT_AXIS))


# A neutral query string that hits NO substring trigger lists in
# memory / rag / web. With this query, all *_score (substring) values
# are 0.0 and the only signal driving routing is the embedding
# component — exactly what most Tier-2 tests want to isolate.
_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"


# ============================================================
# 1. Embedding scorer basics
# ============================================================
class EmbeddingScorerBasicTests(unittest.TestCase):
    """``_score_*_intent_embedding`` returns cosine against the
    installed centroid, and 0.0 when the centroid is not installed
    (back-compat with Tier-1)."""

    def test_unset_centroids_return_zero_embedding_scores(self) -> None:
        router = CognitiveRouterV4()
        v = _intent_vec(m=0.9)  # would score 0.9 against memory IF installed
        self.assertEqual(router._score_memory_intent_embedding(v), 0.0)
        self.assertEqual(router._score_rag_intent_embedding(v), 0.0)
        self.assertEqual(router._score_web_intent_embedding(v), 0.0)

    def test_installed_centroids_return_cosine(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.7, r=0.4, w=0.2)
        self.assertAlmostEqual(
            router._score_memory_intent_embedding(v), 0.7, places=4,
        )
        self.assertAlmostEqual(
            router._score_rag_intent_embedding(v), 0.4, places=4,
        )
        self.assertAlmostEqual(
            router._score_web_intent_embedding(v), 0.2, places=4,
        )

    def test_none_intent_vector_returns_zero(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        self.assertEqual(router._score_memory_intent_embedding(None), 0.0)
        self.assertEqual(router._score_rag_intent_embedding(None), 0.0)
        self.assertEqual(router._score_web_intent_embedding(None), 0.0)

    def test_semantic_phrase_lights_lane_without_substring(self) -> None:
        """End-to-end: with centroids installed and a query that
        matches NO substring trigger, the embedding cosine alone
        must light up the lane and select the matching route."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # m=0.6 cosine on the memory axis only.
        v = _intent_vec(m=0.6)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["memory_score"], 0.0)  # substring missed
        self.assertAlmostEqual(d["memory_score_embedding"], 0.6, places=4)
        self.assertAlmostEqual(d["memory_score_final"], 0.6, places=4)
        self.assertEqual(d["memory_score_source"], "embedding")
        self.assertEqual(d["route"], "memory", str(d))


# ============================================================
# 2. Dual-scoring fusion safety
# ============================================================
class DualScoringFusionTests(unittest.TestCase):
    """``final_score = max(substring, embedding)`` — substring wins
    when stronger, embedding wins when stronger, source telemetry
    reflects which won."""

    def test_substring_wins_when_higher(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # 2 memory triggers => substring 2/7 ≈ 0.286.
        # Embedding cosine = 0.10 (much lower).
        v = _intent_vec(m=0.10)
        d = router.route("remember, did i ever like coffee?", intent_vector=v)
        self.assertGreater(d["memory_score"], d["memory_score_embedding"])
        self.assertAlmostEqual(d["memory_score_final"], d["memory_score"], places=6)
        self.assertEqual(d["memory_score_source"], "substring")

    def test_embedding_wins_when_higher(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # No substring hits; embedding cosine = 0.55 on memory.
        v = _intent_vec(m=0.55)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["memory_score"], 0.0)
        self.assertAlmostEqual(d["memory_score_embedding"], 0.55, places=4)
        self.assertAlmostEqual(d["memory_score_final"], 0.55, places=4)
        self.assertEqual(d["memory_score_source"], "embedding")

    def test_fusion_is_max_not_average(self) -> None:
        """Critical contract: fusion uses ``max`` not an average. A
        future refactor that silently averages would lower precision
        on the substring-dominant case. Pin the contract."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(r=0.05)  # tiny embedding
        # 2 rag triggers ("according to", "in my") -> 2/7 ≈ 0.286.
        d = router.route("according to the doc in my head", intent_vector=v)
        # If averaged: (0.286 + 0.05) / 2 ≈ 0.168; max picks 0.286.
        self.assertAlmostEqual(d["rag_score_final"], d["rag_score"], places=6)
        self.assertGreater(d["rag_score_final"], 0.20)


# ============================================================
# 3. Confidence floor — tightened lane-relative condition
# ============================================================
class ConfidenceFloorTests(unittest.TestCase):
    """The floor downgrade fires only when the top score is BOTH
    below ``MIN_CONFIDENCE_FLOOR`` AND below
    ``dynamic_<top_intent>_threshold + 0.05``. Anchoring the floor to
    the lane threshold prevents embeddings from silently overriding
    the threshold system."""

    def test_floor_does_not_fire_without_centroids(self) -> None:
        """Activation guard: with NO Tier-2 centroid installed, the
        confidence layer is dormant and a weak embedding never
        downgrades a route. (Substring-only behavior is unchanged
        from Tier 1.)"""
        router = CognitiveRouterV4()
        # No Tier-2 centroids installed.
        # 2 memory triggers fire memory at 2/7 ≈ 0.286 > 0.25 base.
        d = router.route("remember, did i like coffee?")
        self.assertEqual(d["route"], "memory", str(d))
        self.assertFalse(d["tier2_active"])

    def test_weak_top_score_downgrades_to_none(self) -> None:
        """Embedding-only weak signal: top_score = 0.27 is below
        floor 0.30 AND below memory_threshold (0.25) + 0.05 = 0.30.
        Both conditions hold => downgrade to 'none'."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.27)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        # The lane gate fires (0.27 > 0.25), so the priority tree
        # would select 'memory'. The confidence floor must downgrade
        # it because 0.27 < 0.30 and 0.27 < 0.25 + 0.05 = 0.30.
        self.assertAlmostEqual(d["memory_score_final"], 0.27, places=4)
        self.assertEqual(d["top_intent"], "memory")
        self.assertAlmostEqual(d["top_score"], 0.27, places=4)
        self.assertEqual(
            d["route"], "none",
            f"Weak top_score must downgrade to 'none'; got {d!r}",
        )

    def test_substring_top_source_bypasses_floor_downgrade(self) -> None:
        """The floor is meant to filter EMBEDDING noise. A
        substring-driven top score that already cleared its lane
        threshold is a high-precision keyword match and must NOT be
        downgraded by the floor — even when it sits in the
        [base, base+0.05] band that would trip the lane buffer.

        This pins the per-lane source telemetry as a real semantic
        gate (not just observability), so a future refactor that
        accidentally drops the substring exemption fails CI.
        """
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # 2 memory triggers ("remember", "did i") → memory_score = 2/7 ≈ 0.286.
        # No intent vector → memory_score_embedding = 0.0.
        # memory_score_final = max(0.286, 0.0) = 0.286 (source = "substring").
        # 0.286 < 0.30 floor AND 0.286 < 0.25+0.05 — both numeric gates
        # would fire, but substring source must shield the route.
        v = _intent_vec()  # all-residual: zero embedding cosines
        d = router.route("remember, did i like coffee?", intent_vector=v)
        self.assertEqual(d["memory_score_source"], "substring")
        self.assertLess(d["top_score"], MIN_CONFIDENCE_FLOOR)
        self.assertEqual(d["top_intent"], "memory")
        self.assertEqual(
            d["route"], "memory",
            "Substring-driven top score must bypass the floor downgrade; "
            f"got {d!r}",
        )

    def test_strong_lane_clearance_blocks_floor_downgrade(self) -> None:
        """If the top score clears its lane threshold by MORE than
        0.05, the lane-relative buffer is satisfied — even when the
        absolute floor is not — and the route must NOT be downgraded.
        This is the critical anchoring property the user requested.

        Setup: artificially lower the memory base threshold to 0.05
        so a top_score of 0.20 clears it by 0.15 > 0.05, while
        remaining < 0.30 floor. The pre-tightening rule would have
        downgraded; the new rule must not.
        """
        router = CognitiveRouterV4()
        router.base_memory_threshold = 0.05
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.20)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertAlmostEqual(d["memory_score_final"], 0.20, places=4)
        self.assertLess(d["top_score"], MIN_CONFIDENCE_FLOOR)
        # 0.20 >= 0.05 + 0.05 = 0.10 → buffer satisfied → no downgrade.
        self.assertEqual(
            d["route"], "memory",
            f"Strong lane clearance must beat the floor's lane buffer; got {d!r}",
        )

    def test_floor_does_not_apply_to_recall_active(self) -> None:
        """``recall_active`` queries are owned by the recall path and
        already gated by ``recall_threshold`` (0.62) + chat margin —
        they bypass the Tier-2 floor."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _install_recall_chat_centroids(router)
        # recall=0.75, chat=0.10, all retrieval lanes near zero.
        v = _intent_vec(rc=0.75, ch=0.10)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertTrue(d["recall_active"], str(d))
        self.assertEqual(d["route"], "hybrid", str(d))

    def test_floor_does_not_downgrade_no_route(self) -> None:
        """When the priority tree already returned 'none', the floor
        layer is a no-op (nothing to downgrade)."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # All scores ~0; no lane fires.
        v = _intent_vec(m=0.05, r=0.05, w=0.05)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["route"], "none")


# ============================================================
# 4. Ambiguity upgrade
# ============================================================
class AmbiguityMarginTests(unittest.TestCase):
    """Single-lane retrieval routes get upgraded to ``hybrid`` when
    the runner-up retrieval lane is within ``AMBIGUITY_MARGIN`` AND
    itself clears ``MIN_CONFIDENCE_FLOOR``."""

    def test_close_rag_memory_upgrades_to_hybrid(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # rag=0.35 (top), memory=0.32 (within 0.10 margin, >= 0.30 floor).
        v = _intent_vec(r=0.35, m=0.32)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["top_intent"], "rag")
        self.assertLess(d["confidence_margin"], AMBIGUITY_MARGIN)
        self.assertGreaterEqual(d["second_best_score"], MIN_CONFIDENCE_FLOOR)
        self.assertEqual(
            d["route"], "hybrid",
            f"Close rag/memory scores must upgrade to hybrid; got {d!r}",
        )

    def test_clear_top_score_does_not_upgrade(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # rag=0.50 dominant, memory=0.10 well below.
        v = _intent_vec(r=0.50, m=0.10)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["top_intent"], "rag")
        self.assertGreaterEqual(d["confidence_margin"], AMBIGUITY_MARGIN)
        self.assertEqual(d["route"], "rag", str(d))

    def test_close_runner_up_below_floor_does_not_upgrade(self) -> None:
        """Ambiguity upgrade requires the runner-up to clear the
        floor too — otherwise we'd be upgrading on noise.

        Setup: top=rag is above its lane threshold (so the priority
        tree picks single-lane 'rag', not 'hybrid'), but runner-up=
        memory is BELOW its lane threshold (so the priority tree
        won't pre-pick hybrid) AND below the 0.30 floor (so the
        ambiguity layer must not upgrade either)."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # rag=0.32 (>0.25 base, top), memory=0.24 (<0.25 base, <0.30 floor).
        # margin=0.08 < 0.10 (close enough to be ambiguous).
        v = _intent_vec(r=0.32, m=0.24)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["top_intent"], "rag")
        self.assertLess(d["confidence_margin"], AMBIGUITY_MARGIN)
        self.assertLess(d["second_best_score"], MIN_CONFIDENCE_FLOOR)
        self.assertEqual(
            d["route"], "rag",
            f"Runner-up below floor must not trigger hybrid; got {d!r}",
        )

    def test_upgrade_only_for_retrieval_pair(self) -> None:
        """Upgrade must only fire when the runner-up is the OTHER
        retrieval lane (rag<->memory). If the runner-up is web, a
        rag/memory hybrid wouldn't actually fuse the runner-up."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # rag=0.35 wins, web=0.32 close, memory near zero.
        # internet_threshold base = 0.20 so web also fires the lane gate;
        # but priority tree picks 'web' first. To isolate the rule we
        # need rag to win the priority tree, so keep web below the
        # internet threshold.
        # Use rag=0.35, web=0.18, memory=0.05 so:
        #   web_enabled = 0.18 > 0.20 ? False → priority skips web
        #   rag_enabled = 0.35 > 0.25 ? True  → route = 'rag'
        # second_best = web=0.18 < 0.30 floor so upgrade short-circuits
        # there too — pin both behaviors.
        v = _intent_vec(r=0.35, w=0.18, m=0.05)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["route"], "rag", str(d))

    def test_upgrade_does_not_apply_to_web_route(self) -> None:
        """Web is never upgraded to hybrid by this rule — internet is
        qualitatively different from disk retrieval."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # web=0.45 wins lane, rag=0.40 close (>= floor, within margin).
        v = _intent_vec(w=0.45, r=0.40)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["route"], "web", str(d))


# ============================================================
# 5. Decision-dict additive contract
# ============================================================
class DecisionDictTier2ContractTests(unittest.TestCase):
    """The Tier-2 fields are present alongside (not in place of) the
    Tier-1 fields, so existing telemetry/UI consumers keep working."""

    _TIER1_KEYS = (
        "route", "drift",
        "memory_score", "rag_score", "web_score",
        "recall_score", "recall_active", "chat_score",
        "recall_margin_over_chat",
        "rag_threshold", "memory_threshold", "internet_threshold",
        "recall_threshold",
        "strategy",
    )

    _TIER2_KEYS = (
        "memory_score_embedding", "rag_score_embedding", "web_score_embedding",
        "memory_score_final", "rag_score_final", "web_score_final",
        "memory_score_source", "rag_score_source", "web_score_source",
        "top_intent", "top_intent_source",
        "top_score", "second_best_score", "confidence_margin",
        "min_confidence_floor", "ambiguity_margin",
        "tier2_active",
    )

    def test_all_tier1_keys_still_present(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.20))
        for key in self._TIER1_KEYS:
            self.assertIn(key, d, f"Tier-1 key {key!r} missing — additive contract broken.")

    def test_all_tier2_keys_present(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.20))
        for key in self._TIER2_KEYS:
            self.assertIn(key, d, f"Tier-2 key {key!r} missing.")

    def test_tier2_active_flag_reflects_centroid_install(self) -> None:
        router_off = CognitiveRouterV4()
        d_off = router_off.route(_NEUTRAL_QUERY)
        self.assertFalse(d_off["tier2_active"])

        router_on = CognitiveRouterV4()
        _install_tier2_centroids(router_on)
        d_on = router_on.route(_NEUTRAL_QUERY)
        self.assertTrue(d_on["tier2_active"])

    def test_score_source_is_embedding_or_substring(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.30, r=0.10))
        self.assertIn(d["memory_score_source"], ("embedding", "substring"))
        self.assertIn(d["rag_score_source"], ("embedding", "substring"))
        self.assertIn(d["web_score_source"], ("embedding", "substring"))

    def test_top_intent_source_present_without_centroids(self) -> None:
        """``top_intent_source`` is an UNCONDITIONAL field — present
        even on Tier-1-only installs (no centroids yet). This makes
        downstream telemetry that conditions on it safe to deploy
        before the centroids are built.
        """
        router = CognitiveRouterV4()  # no centroids
        d = router.route("according to my notes")  # 2 rag triggers
        self.assertIn("top_intent_source", d)
        self.assertEqual(d["top_intent"], "rag")
        # Substring is the only signal possible without centroids.
        self.assertEqual(d["top_intent_source"], "substring")

    def test_top_intent_source_tracks_substring_winner(self) -> None:
        """When the top lane's substring score beat its embedding
        score, ``top_intent_source`` must report 'substring'."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # Memory substring 2/7 ≈ 0.286 > tiny embedding 0.10.
        v = _intent_vec(m=0.10)
        d = router.route("remember, did i like coffee?", intent_vector=v)
        self.assertEqual(d["top_intent"], "memory")
        self.assertEqual(d["memory_score_source"], "substring")
        self.assertEqual(d["top_intent_source"], "substring")

    def test_top_intent_source_tracks_embedding_winner(self) -> None:
        """When the top lane's embedding score beat its substring
        score, ``top_intent_source`` must report 'embedding'."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # No substring hits; embedding cosine = 0.55 on memory.
        v = _intent_vec(m=0.55)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["top_intent"], "memory")
        self.assertEqual(d["memory_score_source"], "embedding")
        self.assertEqual(d["top_intent_source"], "embedding")

    def test_top_intent_source_for_recall_is_substring(self) -> None:
        """The recall lane is gated separately (recall_threshold +
        chat margin), not by the Tier-2 floor. The convention is
        that ``top_intent_source == 'substring'`` whenever recall is
        the top intent — pinning the contract."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _install_recall_chat_centroids(router)
        v = _intent_vec(rc=0.75, ch=0.10)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["top_intent"], "recall")
        self.assertEqual(d["top_intent_source"], "substring")

    def test_confidence_constants_match_module_constants(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.20))
        self.assertAlmostEqual(d["min_confidence_floor"], MIN_CONFIDENCE_FLOOR, places=6)
        self.assertAlmostEqual(d["ambiguity_margin"], AMBIGUITY_MARGIN, places=6)


# ============================================================
# 6. Regression guard — explicit substring queries still route.
# ============================================================
class RegressionGuardTests(unittest.TestCase):
    """The Tier-2 layer must not regress any of the Tier-1 substring
    routing behaviors. We re-run a few of the substring scenarios
    from ``test_cognitive_router_margin.py`` with the Tier-2
    centroids installed."""

    def test_explicit_web_query_still_routes_to_web(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # 4 web triggers: "search the web", "today", "right now", "weather"
        d = router.route(
            "search the web for today's weather right now",
            intent_vector=_intent_vec(),
        )
        # web substring score = 4/13 ≈ 0.308 > 0.20 base.
        self.assertGreater(d["web_score"], router.base_internet_threshold)
        self.assertEqual(d["route"], "web", str(d))

    def test_explicit_memory_query_still_routes_to_memory(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(
            "remember, did i like coffee?",
            intent_vector=_intent_vec(),
        )
        self.assertGreater(d["memory_score"], router.base_memory_threshold)
        self.assertIn(d["route"], ("memory", "hybrid"), str(d))

    def test_explicit_rag_query_still_routes_to_rag(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(
            "according to my notes",
            intent_vector=_intent_vec(),
        )
        self.assertGreater(d["rag_score"], router.base_rag_threshold)
        self.assertIn(d["route"], ("rag", "hybrid"), str(d))

    def test_no_centroids_no_intent_vector_unchanged_routing(self) -> None:
        """The Tier-1 path is fully exercised — and unchanged — when
        no centroids and no intent vector are provided."""
        router = CognitiveRouterV4()  # no centroids
        d = router.route("according to my notes")  # no intent_vector
        self.assertFalse(d["tier2_active"])
        self.assertEqual(d["memory_score_embedding"], 0.0)
        self.assertEqual(d["rag_score_embedding"], 0.0)
        self.assertEqual(d["web_score_embedding"], 0.0)
        self.assertIn(d["route"], ("rag", "hybrid"))


if __name__ == "__main__":
    unittest.main()
