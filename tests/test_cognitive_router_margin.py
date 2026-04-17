"""Unit tests for the T4.2 cognitive router tightening —
``CognitiveRouterV4`` chat-class negative centroid + margin rule +
raised recall threshold (0.55 -> 0.62).

These tests deliberately avoid importing the embedder / Qt / audio
stack: ``CognitiveRouterV4`` is pure numpy, and we exercise it
directly with synthetic unit vectors whose cosine against each
centroid is controlled analytically. The single llm_worker
contract test parses the source text (same pattern as
``test_memory_tier_routing.LLMWorkerSourceContractTests`` and
``test_rag_relevance_gate.LLMWorkerDowngradeContractTests``) so we
don't pull in Qt during test collection.
"""
from __future__ import annotations

import math
import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.cognitive_router import CognitiveRouterV4


# ============================================================
# Synthetic vector helpers.
#
# We work in R^3 because two orthogonal centroid axes + one
# "residual" axis is the minimum geometry we need to hit any
# (recall_cos, chat_cos) pair with a single unit intent vector.
# Orthogonal centroids make the cosine-vs-each-centroid math
# equal to picking the first two coordinates of a unit vector.
# ============================================================
_RECALL_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_CHAT_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _intent_with_cosines(recall_cos: float, chat_cos: float) -> np.ndarray:
    """Return a unit vector ``v`` with
    ``np.dot(v, _RECALL_AXIS) == recall_cos`` and
    ``np.dot(v, _CHAT_AXIS) == chat_cos``.

    Because the centroids are orthogonal, the residual component on
    the third axis is ``sqrt(1 - recall_cos^2 - chat_cos^2)``, which
    is real as long as the caller picks compatible values (we only
    ever use small positive cosines, so this is trivially safe).
    """
    rsq = recall_cos * recall_cos + chat_cos * chat_cos
    if rsq > 1.0:
        raise ValueError(
            f"Incompatible cosine targets: recall={recall_cos}, chat={chat_cos} "
            f"(sum of squares {rsq:.4f} > 1.0 — no unit vector exists)."
        )
    residual = math.sqrt(max(0.0, 1.0 - rsq))
    v = np.array([recall_cos, chat_cos, residual], dtype=np.float32)
    # Guard against floating-point norm drift from the sqrt above.
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v


def _install_both_centroids(router: CognitiveRouterV4) -> None:
    router.set_recall_centroid(_RECALL_AXIS.copy())
    router.set_chat_centroid(_CHAT_AXIS.copy())


# Query strings — deliberately chosen so the substring fallback in
# ``detect_recall_intent`` does NOT fire. That way the cosine path
# is the only thing driving ``recall_score``, and test assertions
# are pinned to the math we constructed rather than to incidental
# word matches. (The dedicated substring-fallback test below opts
# back in to the fallback on purpose.)
_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"


class CognitiveRouterMarginTests(unittest.TestCase):
    # ----------------------------------------------------------
    # Constant guards — the thresholds are the whole point of
    # this PR and a future refactor that silently loosens them
    # should fail CI.
    # ----------------------------------------------------------
    def test_recall_threshold_is_0_62(self) -> None:
        """T4.2 bumps ``recall_threshold`` from 0.55 -> 0.62."""
        router = CognitiveRouterV4()
        self.assertAlmostEqual(router.recall_threshold, 0.62, places=6)

    def test_recall_margin_over_chat_is_0_05(self) -> None:
        """The margin constant stays at 0.05 — only the mechanism
        using it changed."""
        router = CognitiveRouterV4()
        self.assertAlmostEqual(router.recall_margin_over_chat, 0.05, places=6)

    # ----------------------------------------------------------
    # 1. Margin rule blocks false recall: recall >= threshold AND
    #    chat ~= recall => margin < 0.05 => recall_active False.
    # ----------------------------------------------------------
    def test_margin_rule_blocks_recall_when_chat_is_equally_close(self) -> None:
        """When a query is equally close to both centroids, the
        margin gate must block ``recall_active`` even though the
        absolute threshold is satisfied. This is the sky-blue
        regression being fixed in T4.2."""
        router = CognitiveRouterV4()
        _install_both_centroids(router)
        # recall = chat = 0.65 >= 0.62, margin = 0.0 < 0.05
        intent = _intent_with_cosines(0.65, 0.65)
        decision = router.route(_NEUTRAL_QUERY, intent_vector=intent)
        self.assertAlmostEqual(decision["recall_score"], 0.65, places=4)
        self.assertAlmostEqual(decision["chat_score"], 0.65, places=4)
        self.assertFalse(
            decision["recall_active"],
            f"Recall should be blocked by margin rule, got: {decision}",
        )
        self.assertNotEqual(
            decision["route"], "hybrid",
            "A blocked-by-margin recall must not force the hybrid route.",
        )

    # ----------------------------------------------------------
    # 2. Margin rule lets true recall through.
    # ----------------------------------------------------------
    def test_margin_rule_allows_recall_with_sufficient_margin(self) -> None:
        """When ``recall - chat >= 0.05`` and ``recall >= 0.62``, the
        router must set ``recall_active = True`` and upgrade the
        route to ``hybrid``."""
        router = CognitiveRouterV4()
        _install_both_centroids(router)
        # recall = 0.75, chat = 0.60, margin = 0.15
        intent = _intent_with_cosines(0.75, 0.60)
        decision = router.route(_NEUTRAL_QUERY, intent_vector=intent)
        self.assertAlmostEqual(decision["recall_score"], 0.75, places=4)
        self.assertAlmostEqual(decision["chat_score"], 0.60, places=4)
        self.assertTrue(decision["recall_active"], str(decision))
        self.assertEqual(decision["route"], "hybrid", str(decision))

    # ----------------------------------------------------------
    # 3. Backwards compatibility: chat centroid unset => margin
    #    gate trivially satisfied (chat_score defaults to 0.0).
    # ----------------------------------------------------------
    def test_backwards_compat_without_chat_centroid(self) -> None:
        """Pre-T4.2 installs only install the recall centroid. The
        router must still fire ``recall_active`` on a strong recall
        score because ``chat_score`` defaults to 0.0 and the margin
        (recall - 0) is trivially satisfied."""
        router = CognitiveRouterV4()
        # Deliberately omit set_chat_centroid.
        router.set_recall_centroid(_RECALL_AXIS.copy())
        self.assertIsNone(router.chat_centroid)
        intent = _intent_with_cosines(0.75, 0.0)
        decision = router.route(_NEUTRAL_QUERY, intent_vector=intent)
        self.assertAlmostEqual(decision["recall_score"], 0.75, places=4)
        self.assertEqual(decision["chat_score"], 0.0)
        self.assertTrue(decision["recall_active"], str(decision))
        self.assertEqual(decision["route"], "hybrid", str(decision))

    # ----------------------------------------------------------
    # 4. Threshold bump is observable — 0.58 used to fire at the
    #    old 0.55 floor, must not fire at the new 0.62 floor.
    # ----------------------------------------------------------
    def test_threshold_bump_blocks_score_below_0_62(self) -> None:
        """A recall score of 0.58 sat comfortably above the old
        0.55 threshold and was a big source of the false-HYBRID
        routing. At the new 0.62 floor, it must not fire —
        independently of the margin rule."""
        router = CognitiveRouterV4()
        _install_both_centroids(router)
        # Keep chat low so we know it's the threshold doing the
        # blocking, not the margin gate.
        intent = _intent_with_cosines(0.58, 0.10)
        decision = router.route(_NEUTRAL_QUERY, intent_vector=intent)
        self.assertAlmostEqual(decision["recall_score"], 0.58, places=4)
        self.assertFalse(
            decision["recall_active"],
            f"recall_score=0.58 must NOT fire recall at the new 0.62 "
            f"threshold, got: {decision}",
        )
        self.assertNotEqual(decision["route"], "hybrid")

    # ----------------------------------------------------------
    # 5. Substring fallback still works when no intent vector is
    #    available (embedder-less installs, early boot, etc.).
    # ----------------------------------------------------------
    def test_substring_fallback_still_fires_recall(self) -> None:
        """When ``intent_vector`` is ``None`` and the query contains
        a recall phrase like "do you remember", ``_score_recall_intent``
        falls back to 1.0 via ``detect_recall_intent`` and
        ``_score_chat_intent`` returns 0.0 (no fallback on the
        negative class). Margin = 1.0 >= 0.05, threshold = 1.0 >=
        0.62, so ``recall_active`` must be True."""
        router = CognitiveRouterV4()
        _install_both_centroids(router)
        decision = router.route(
            "do you remember my favorite color?",
            intent_vector=None,
        )
        self.assertAlmostEqual(decision["recall_score"], 1.0, places=4)
        self.assertEqual(decision["chat_score"], 0.0)
        self.assertTrue(decision["recall_active"], str(decision))
        self.assertEqual(decision["route"], "hybrid", str(decision))

    # ----------------------------------------------------------
    # Extra: decision dict contract — the new fields must appear
    # alongside the pre-existing ones, and no existing field may
    # be renamed (router_telemetry / tuner consumers depend on
    # the existing keys).
    # ----------------------------------------------------------
    def test_decision_dict_exposes_new_and_old_fields(self) -> None:
        router = CognitiveRouterV4()
        _install_both_centroids(router)
        decision = router.route(
            _NEUTRAL_QUERY,
            intent_vector=_intent_with_cosines(0.30, 0.30),
        )
        for key in (
            "route", "recall_score", "recall_active", "recall_threshold",
            "chat_score", "recall_margin_over_chat",
            "memory_score", "rag_score", "web_score",
        ):
            self.assertIn(key, decision, f"decision dict missing {key!r}")


class LLMWorkerChatCentroidContractTests(unittest.TestCase):
    """Static check that ``workers/llm_worker.py`` wires the chat
    centroid symmetrically to the recall centroid: defines
    ``_CHAT_INTENT_EXAMPLES``, calls ``set_chat_centroid(...)``, and
    guards the install behind a ``chat_centroid is None`` check so
    we don't rebuild on every turn.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "workers", "llm_worker.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_chat_intent_examples_is_defined(self) -> None:
        self.assertRegex(
            self.src,
            r"_CHAT_INTENT_EXAMPLES\s*=\s*\(",
            "llm_worker must define _CHAT_INTENT_EXAMPLES for the "
            "negative-class centroid.",
        )

    def test_set_chat_centroid_is_called(self) -> None:
        self.assertIn(
            "set_chat_centroid(",
            self.src,
            "llm_worker must install the chat centroid via "
            "cognitive_router.set_chat_centroid(...).",
        )

    def test_chat_centroid_install_is_guarded_by_is_none_check(self) -> None:
        """The install must be guarded by a ``chat_centroid is None``
        test so we don't rebuild the centroid on every turn (and so
        manual overrides via ``set_chat_centroid`` aren't stomped
        back to the curated default)."""
        match = re.search(
            r"if\s+self\.cognitive_router\.chat_centroid\s+is\s+None\s*:"
            r"\s*\n[^}]*?set_chat_centroid\(",
            self.src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "Expected ``if self.cognitive_router.chat_centroid is None:`` "
            "guarding the ``set_chat_centroid(...)`` call so the centroid "
            "is built at most once per worker lifetime.",
        )


if __name__ == "__main__":
    unittest.main()
