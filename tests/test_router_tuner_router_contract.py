"""Contract tests between ``AdaptiveRouterSelfTunerV2.get_weights()``
and ``CognitiveRouterV4.route()``.

The original Tier-1 bug was a silent key mismatch: the tuner emitted
``{"hybrid", "memory", "rag"}`` while the router read
``weights.get("rag_sensitivity" / "memory_sensitivity" /
"internet_sensitivity", 1.0)``. Every read missed → defaulted to 1.0
→ adaptive multiplier was always 1.0 → tuning had zero behavioral
effect.

These tests pin the contract from both sides:

  1. The tuner produces all three sensitivity keys the router consumes.
  2. The router source still reads exactly those key names (catches a
     future router-side rename that would silently re-orphan the tuner).
  3. End-to-end behavioral check: a non-trivial sensitivity actually
     moves the threshold reported in the decision dict.
  4. ``internet_sensitivity`` exists on the tuner at the neutral
     default of 1.0 (no _adjust rule consumes it yet — that PR comes
     later, but the field has to exist now so the router has a real
     source value).

All tests are pure-Python — no Qt, no embedder, no MCP imports.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.cognitive_router import CognitiveRouterV4
from mcp.router_self_tuner import AdaptiveRouterSelfTunerV2


# Exact sensitivity key names exchanged across the boundary. Pinning
# these here means a rename on either side fails this file before it
# silently disconnects the tuner.
_REQUIRED_SENSITIVITY_KEYS = frozenset({
    "rag_sensitivity",
    "memory_sensitivity",
    "internet_sensitivity",
})


class TunerEmitsKeysRouterReadsTests(unittest.TestCase):
    def test_tuner_emits_all_router_keys(self) -> None:
        """The tuner's ``get_weights()`` output MUST include every
        key the router reads; otherwise the router silently falls
        back to its 1.0 defaults and self-tuning is a no-op."""
        weights = AdaptiveRouterSelfTunerV2().get_weights()
        missing = _REQUIRED_SENSITIVITY_KEYS - set(weights.keys())
        self.assertFalse(
            missing,
            f"Tuner.get_weights() is missing router-consumed keys: "
            f"{sorted(missing)}. Full payload: {weights!r}",
        )

    def test_internet_sensitivity_default_is_one(self) -> None:
        """``internet_sensitivity`` rides at the neutral default of
        1.0 in this PR — no ``_adjust`` rule consumes it yet. The
        field exists so the router has a real source value instead
        of always falling back to its own 1.0 default."""
        tuner = AdaptiveRouterSelfTunerV2()
        self.assertEqual(tuner.internet_sensitivity, 1.0)
        self.assertEqual(tuner.get_weights()["internet_sensitivity"], 1.0)

    def test_get_weights_values_are_floats(self) -> None:
        """Defensive: every value the router will multiply must be a
        plain float so the multiplier math does not crash."""
        weights = AdaptiveRouterSelfTunerV2().get_weights()
        for key, value in weights.items():
            self.assertIsInstance(
                value, float,
                f"weights[{key!r}] is {type(value).__name__}, expected float",
            )


class RouterSourceReadsSharedKeysTests(unittest.TestCase):
    """Static-source check that the router still reads the exact
    sensitivity key names this contract requires. Catches a future
    router-side rename that would silently break the wires the
    other direction."""

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "mcp", "cognitive_router.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_router_reads_rag_sensitivity(self) -> None:
        self.assertIn(
            '"rag_sensitivity"',
            self.src,
            "cognitive_router.py must reference the literal "
            "'rag_sensitivity' key consumed from tuner weights.",
        )

    def test_router_reads_memory_sensitivity(self) -> None:
        self.assertIn(
            '"memory_sensitivity"',
            self.src,
            "cognitive_router.py must reference the literal "
            "'memory_sensitivity' key consumed from tuner weights.",
        )

    def test_router_reads_internet_sensitivity(self) -> None:
        self.assertIn(
            '"internet_sensitivity"',
            self.src,
            "cognitive_router.py must reference the literal "
            "'internet_sensitivity' key consumed from tuner weights.",
        )


class RouterConsumesSharedKeysBehaviorallyTests(unittest.TestCase):
    """End-to-end check that the wires are connected: passing
    non-trivial sensitivities through ``route(...)`` measurably moves
    the thresholds reported in the decision dict."""

    _NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"

    def test_high_rag_sensitivity_lowers_rag_threshold(self) -> None:
        router = CognitiveRouterV4()
        decision = router.route(
            self._NEUTRAL_QUERY,
            weights={"rag_sensitivity": 1.5},
        )
        self.assertLess(
            decision["rag_threshold"],
            router.base_rag_threshold,
            f"rag_sensitivity=1.5 should LOWER rag_threshold below "
            f"base ({router.base_rag_threshold}); got "
            f"{decision['rag_threshold']}.",
        )

    def test_low_memory_sensitivity_raises_memory_threshold(self) -> None:
        router = CognitiveRouterV4()
        decision = router.route(
            self._NEUTRAL_QUERY,
            weights={"memory_sensitivity": 0.6},
        )
        self.assertGreater(
            decision["memory_threshold"],
            router.base_memory_threshold,
            f"memory_sensitivity=0.6 should RAISE memory_threshold "
            f"above base ({router.base_memory_threshold}); got "
            f"{decision['memory_threshold']}.",
        )

    def test_no_weights_yields_base_thresholds(self) -> None:
        """Sanity baseline: no weights -> dynamic thresholds equal
        the bases (no latency samples at startup, no offset)."""
        router = CognitiveRouterV4()
        decision = router.route(self._NEUTRAL_QUERY, weights=None)
        self.assertAlmostEqual(
            decision["rag_threshold"], router.base_rag_threshold, places=6,
        )
        self.assertAlmostEqual(
            decision["memory_threshold"], router.base_memory_threshold, places=6,
        )
        self.assertAlmostEqual(
            decision["internet_threshold"],
            router.base_internet_threshold,
            places=6,
        )

    def test_tuner_to_router_end_to_end(self) -> None:
        """The full producer -> consumer wire: feed the tuner's
        actual ``get_weights()`` output into the router and confirm
        it does not crash and produces a valid decision dict. This is
        the integration assert that would have caught the original
        bug regardless of which side did the renaming."""
        tuner = AdaptiveRouterSelfTunerV2()
        router = CognitiveRouterV4()
        decision = router.route(
            self._NEUTRAL_QUERY,
            weights=tuner.get_weights(),
        )
        for key in (
            "route", "rag_threshold", "memory_threshold", "internet_threshold",
        ):
            self.assertIn(key, decision)


if __name__ == "__main__":
    unittest.main()
