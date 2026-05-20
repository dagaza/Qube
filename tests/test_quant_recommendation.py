"""Tests for quantization recommendation heuristics."""

from __future__ import annotations

import unittest

from core.quant_recommendation import (
    QuantBadgeKind,
    RecommendationConfidence,
    apply_tool_calling_modifier,
    build_context_from_hub_meta,
    recommend_quants,
    resolve_size_band,
)


def _files(*paths: str) -> list[tuple[str, int | None]]:
    sizes = {
        "a-Q8_0.gguf": 4_000_000_000,
        "a-Q6_K.gguf": 3_200_000_000,
        "a-Q5_K_M.gguf": 5_000_000_000,
        "a-Q4_K_M.gguf": 20_000_000_000,
    }
    return [(p, sizes.get(p)) for p in paths]


class TestQuantRecommendation(unittest.TestCase):
    def test_3b_tiny_band(self) -> None:
        r = resolve_size_band(3.0)
        self.assertEqual(r.band.band_id, "tiny")
        self.assertEqual(r.band.primary, "Q8_0")

    def test_8b_medium_band(self) -> None:
        r = resolve_size_band(8.0)
        self.assertEqual(r.band.band_id, "medium")
        self.assertEqual(r.band.primary, "Q5_K_M")

    def test_20b_large_mid_band(self) -> None:
        r = resolve_size_band(20.0)
        self.assertEqual(r.band.band_id, "large_mid")
        self.assertEqual(r.band.primary, "Q5_K_M")

    def test_34b_xlarge_band(self) -> None:
        r = resolve_size_band(34.0)
        self.assertEqual(r.band.band_id, "xlarge")
        self.assertEqual(r.band.primary, "Q4_K_M")

    def test_unknown_band_defaults_q5(self) -> None:
        r = resolve_size_band(None)
        self.assertEqual(r.band.band_id, "unknown")
        self.assertEqual(r.band.primary, "Q5_K_M")

    def test_recommend_8b_defaults_q5(self) -> None:
        ctx = build_context_from_hub_meta(
            repo_id="org/model-8b",
            title="Model 8B",
            description="",
            meta={"params": "8B"},
        )
        plan = recommend_quants(ctx, _files("a-Q5_K_M.gguf", "a-Q4_K_M.gguf", "a-Q6_K.gguf"))
        self.assertEqual(plan.primary_quant, "Q5_K_M")
        self.assertEqual(plan.default_index, 1)
        rec0 = plan.files[0]
        self.assertEqual(rec0.badge, QuantBadgeKind.RECOMMENDED)

    def test_tiny_tool_modifier_prefers_q5_when_present(self) -> None:
        ctx = build_context_from_hub_meta(
            repo_id="org/tool-3b",
            title="Tool 3B",
            description="function calling",
            meta={"params": "3B", "capabilities": ["Tool Use"]},
        )
        self.assertGreaterEqual(ctx.tool_calling_score, 0.35)
        plan = recommend_quants(
            ctx,
            _files("a-Q8_0.gguf", "a-Q5_K_M.gguf", "a-Q6_K.gguf"),
        )
        self.assertEqual(plan.band_id, "tiny")
        default = plan.files[plan.default_index - 1] if plan.default_index else None
        self.assertIsNotNone(default)
        assert default is not None
        self.assertEqual(default.quant_label, "Q5_K_M")

    def test_34b_tool_keeps_q4_primary_badge(self) -> None:
        ctx = build_context_from_hub_meta(
            repo_id="org/big-tool",
            title="Tool 70B",
            description="",
            meta={"params": "34B"},
        )
        ctx = build_context_from_hub_meta(
            repo_id="org/big-tool",
            title="Tool",
            description="function calling agent",
            meta={"params": "34B"},
        )
        mod = apply_tool_calling_modifier(resolve_size_band(34.0).band, ctx.tool_calling_score)
        self.assertGreater(mod.q5_bonus, 0.0)
        plan = recommend_quants(ctx, _files("a-Q4_K_M.gguf", "a-Q5_K_M.gguf"))
        q4 = next(f for f in plan.files if f.quant_label == "Q4_K_M")
        self.assertEqual(q4.badge, QuantBadgeKind.RECOMMENDED)

    def test_secondary_badge_lower_memory_on_tiny(self) -> None:
        ctx = build_context_from_hub_meta(
            repo_id="x",
            title="3B",
            description="",
            meta={"params": "3B"},
        )
        plan = recommend_quants(ctx, _files("a-Q8_0.gguf", "a-Q6_K.gguf"))
        q6 = next(f for f in plan.files if f.quant_label == "Q6_K")
        self.assertEqual(q6.badge, QuantBadgeKind.LOWER_MEMORY)

    def test_confidence_unknown_params_low(self) -> None:
        ctx = build_context_from_hub_meta(
            repo_id="mystery/model",
            title="Mystery",
            description="no size here",
            meta={},
        )
        plan = recommend_quants(ctx, _files("a-Q5_K_M.gguf"))
        self.assertEqual(plan.plan_confidence, RecommendationConfidence.LOW)

    def test_confidence_hf_card_high(self) -> None:
        ctx = build_context_from_hub_meta(
            repo_id="org/m",
            title="M",
            description="",
            meta={"params": "7B"},
        )
        self.assertEqual(ctx.params_source, "hf_card")
        plan = recommend_quants(ctx, _files("a-Q5_K_M.gguf"))
        self.assertEqual(plan.plan_confidence, RecommendationConfidence.HIGH)


if __name__ == "__main__":
    unittest.main()
