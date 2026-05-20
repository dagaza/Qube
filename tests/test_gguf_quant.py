"""Tests for GGUF quant parsing."""

from __future__ import annotations

import unittest

from core.gguf_quant import (
    QuantFamily,
    parse_quant_from_gguf_path,
    parse_quant_token,
    quant_matches,
    rank_distance_to_preferred,
)


class TestGgufQuant(unittest.TestCase):
    def test_parse_q5_k_m_path(self) -> None:
        p = parse_quant_from_gguf_path("org/Model-7B-Q5_K_M.gguf")
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.normalized, "Q5_K_M")
        self.assertEqual(p.family, QuantFamily.K_QUANT)

    def test_parse_dash_variant(self) -> None:
        p = parse_quant_from_gguf_path("model-q5-k-m.gguf")
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.normalized, "Q5_K_M")

    def test_parse_iq4_xs(self) -> None:
        p = parse_quant_from_gguf_path("llama-IQ4_XS.gguf")
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.family, QuantFamily.IQ_QUANT)
        self.assertEqual(p.normalized, "IQ4_XS")

    def test_quant_matches_same_family(self) -> None:
        self.assertTrue(quant_matches("Q5_K_M", "Q5-K-M"))
        self.assertFalse(quant_matches("Q5_K_M", "IQ4_XS"))

    def test_iq_does_not_match_k(self) -> None:
        self.assertFalse(quant_matches("Q4_K_M", "IQ4_XS"))

    def test_rank_distance(self) -> None:
        p = parse_quant_token("Q4_K_M")
        self.assertEqual(rank_distance_to_preferred(p, "Q5_K_M"), 10)


if __name__ == "__main__":
    unittest.main()
