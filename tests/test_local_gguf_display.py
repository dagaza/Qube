"""Tests for local GGUF display-name formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.local_gguf_display import format_local_gguf_display, local_gguf_sort_key
from core.model_params import parse_params_b_from_filename


class TestLocalGgufDisplay(unittest.TestCase):
    def test_standard_quant_path(self) -> None:
        disp = format_local_gguf_display("/tmp/Model-7B-Q5_K_M.gguf")
        self.assertEqual(disp.basename, "Model-7B-Q5_K_M.gguf")
        self.assertEqual(disp.menu_label, "Model-7B · Q5_K_M")
        self.assertEqual(disp.button_label, "Model-7B · Q5_K_M")
        self.assertTrue(disp.tooltip.endswith("Model-7B-Q5_K_M.gguf"))

    def test_dash_variant_quant(self) -> None:
        disp = format_local_gguf_display("/models/model-q5-k-m.gguf")
        self.assertEqual(disp.menu_label, "model · Q5_K_M")

    def test_iq_quant(self) -> None:
        disp = format_local_gguf_display("/models/llama-IQ4_XS.gguf")
        self.assertEqual(disp.menu_label, "llama · IQ4_XS")

    def test_no_quant_fallback(self) -> None:
        disp = format_local_gguf_display("/models/custom-model.gguf")
        self.assertEqual(disp.menu_label, "custom-model.gguf")
        self.assertEqual(disp.button_label, "custom-model.gguf")

    def test_sharded_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "BigModel-7B"
            names = [
                f"{prefix}-00001-of-00003.gguf",
                f"{prefix}-00002-of-00003.gguf",
            ]
            for name in names:
                (root / name).write_bytes(b"x")
            path = root / names[0]
            disp = format_local_gguf_display(str(path), models_dir=root)
            self.assertEqual(disp.menu_label, "BigModel-7B.gguf (2/3 shards)")

    def test_parse_params_b_from_filename(self) -> None:
        self.assertEqual(parse_params_b_from_filename("Qwen2.5-7B-Instruct-Q4_K_M.gguf"), 7.0)
        self.assertEqual(parse_params_b_from_filename("llama-3.2-3B-Instruct-Q4_K_M.gguf"), 3.0)
        self.assertEqual(parse_params_b_from_filename("Model-7B-Q5_K_M.gguf"), 7.0)
        self.assertIsNone(parse_params_b_from_filename("custom-model.gguf"))

    def test_local_gguf_sort_key_ascending(self) -> None:
        names = [
            "Big-70B-Q4_K_M.gguf",
            "Small-4B-Q4_K_M.gguf",
            "Mid-7B-Q4_K_M.gguf",
            "unknown.gguf",
        ]
        ordered = sorted(names, key=local_gguf_sort_key)
        self.assertEqual(
            ordered,
            ["Small-4B-Q4_K_M.gguf", "Mid-7B-Q4_K_M.gguf", "Big-70B-Q4_K_M.gguf", "unknown.gguf"],
        )


if __name__ == "__main__":
    unittest.main()
