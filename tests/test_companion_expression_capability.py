"""Tests for expression capability tier routing."""

from __future__ import annotations

import unittest
from unittest import mock

from core.companion_cognition.capability import (
    ExpressionCapabilityTier,
    allows_full_generate,
    allows_sidecar_rewrite,
    max_expression_level,
    resolve_expression_capability,
)
from core.companion_cognition.types import ExpressionLevel


class TestCompanionExpressionCapability(unittest.TestCase):
    def test_qwen05_maps_micro(self) -> None:
        tier = resolve_expression_capability(sidecar_basename="Qwen2-0.5B-Instruct-Q4_K_M.gguf")
        self.assertEqual(tier, ExpressionCapabilityTier.MICRO)
        self.assertEqual(max_expression_level(tier), ExpressionLevel.TEMPLATE)
        self.assertFalse(allows_sidecar_rewrite(tier))

    def test_large_allows_l3_on_test_with_expressive(self) -> None:
        with mock.patch(
            "core.app_settings.get_companion_expression_freedom",
            return_value="expressive",
        ):
            tier = resolve_expression_capability(sidecar_basename="Llama-3.1-70B-Q4_K_M.gguf")
            self.assertTrue(allows_full_generate(tier, trigger="test"))


if __name__ == "__main__":
    unittest.main()
