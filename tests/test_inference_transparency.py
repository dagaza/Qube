"""Tests for inference transparency helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from core.inference_transparency import (
    describe_layer_configuration,
    format_transparency_lines,
    format_transparency_rows,
    merge_native_telemetry_snapshot,
    normalize_requested_layers,
    parse_backend_hint,
    _static_build_snapshot,
)


class TestParseBackendHint(unittest.TestCase):
    def test_cuda(self) -> None:
        info = "AVX = 1 | CUDA = 1 | OPENMP = 1"
        self.assertEqual(parse_backend_hint(info), "cuda")

    def test_vulkan(self) -> None:
        info = "AVX2 = 1 | Vulkan = 1"
        self.assertEqual(parse_backend_hint(info), "vulkan")

    def test_cpu_fallback(self) -> None:
        self.assertEqual(parse_backend_hint("AVX = 1 | OPENMP = 1"), "cpu")


class TestNormalizeRequestedLayers(unittest.TestCase):
    def test_zero_is_cpu(self) -> None:
        n, label = normalize_requested_layers(0, 32)
        self.assertEqual(n, 0)
        self.assertIn("CPU", label)

    def test_all_layers_sentinel(self) -> None:
        n, label = normalize_requested_layers(0x7FFFFFFF, 33)
        self.assertEqual(n, 33)
        self.assertIn("all", label)

    def test_partial(self) -> None:
        n, label = normalize_requested_layers(16, 32)
        self.assertEqual(n, 16)
        self.assertEqual(label, "16")


class TestDescribeLayerConfiguration(unittest.TestCase):
    def test_no_gpu_build(self) -> None:
        txt = describe_layer_configuration(
            requested_n_gpu_layers=20,
            model_n_layers=32,
            supports_gpu_offload=False,
        )
        self.assertIn("no GPU offload", txt)

    def test_cpu_requested(self) -> None:
        txt = describe_layer_configuration(
            requested_n_gpu_layers=0,
            model_n_layers=32,
            supports_gpu_offload=True,
        )
        self.assertIn("CPU only", txt)

    def test_gpu_requested(self) -> None:
        txt = describe_layer_configuration(
            requested_n_gpu_layers=20,
            model_n_layers=32,
            supports_gpu_offload=True,
        )
        self.assertIn("20", txt)
        self.assertIn("32", txt)


class TestMergeNativeTelemetrySnapshot(unittest.TestCase):
    @mock.patch("core.inference_transparency.get_settings_snapshot")
    @mock.patch("core.inference_transparency.get_hardware_profile_snapshot")
    @mock.patch("core.inference_transparency.get_build_snapshot")
    def test_merges_loaded_native(
        self,
        mock_build: mock.Mock,
        mock_hw: mock.Mock,
        mock_settings: mock.Mock,
    ) -> None:
        mock_build.return_value = {
            "backend_hint": "vulkan",
            "supports_gpu_offload": True,
            "llama_cpp_python_version": "0.3.29",
        }
        mock_hw.return_value = {
            "gpu_memory_kind": "amd_unified",
            "gpu_memory_kind_label": "AMD APU (unified system memory)",
            "is_unified_gpu_memory": True,
            "vram_budget_gb": 16.0,
            "max_safe_n_gpu_layers": 70,
        }
        mock_settings.return_value = {
            "engine_mode": "internal",
            "n_gpu_layers": 33,
            "n_threads": 8,
        }
        native = {
            "loaded": True,
            "requested_n_gpu_layers_normalized": 33,
            "model_n_layers": 33,
            "model_basename": "test.gguf",
        }
        merged = merge_native_telemetry_snapshot(native)
        self.assertTrue(merged["native"]["loaded"])
        self.assertEqual(merged["hardware"]["gpu_memory_kind"], "amd_unified")
        self.assertTrue(merged["settings_match_loaded_layers"])

    @mock.patch("core.inference_transparency.get_settings_snapshot")
    @mock.patch("core.inference_transparency.get_hardware_profile_snapshot")
    @mock.patch("core.inference_transparency.get_build_snapshot")
    @mock.patch("core.inference_transparency._static_build_snapshot")
    def test_unloaded_native_uses_static_build_snapshot(
        self,
        mock_static: mock.Mock,
        mock_build: mock.Mock,
        mock_hw: mock.Mock,
        mock_settings: mock.Mock,
    ) -> None:
        mock_static.return_value = {
            "backend_hint": "cuda",
            "supports_gpu_offload": True,
            "probe_deferred": True,
        }
        mock_hw.return_value = {"gpu_memory_kind": "none"}
        mock_settings.return_value = {"engine_mode": "internal", "n_gpu_layers": 0, "n_threads": 4}

        merged = merge_native_telemetry_snapshot(None)

        mock_build.assert_not_called()
        mock_static.assert_called_once()
        self.assertEqual(merged["build"]["backend_hint"], "cuda")
        self.assertFalse(merged["native"]["loaded"])


class TestStaticBuildSnapshot(unittest.TestCase):
    def test_variant_marker_maps_cuda(self) -> None:
        with mock.patch(
            "core.inference_transparency._read_windows_variant_marker",
            return_value="cuda",
        ):
            snap = _static_build_snapshot()
        self.assertEqual(snap["backend_hint"], "cuda")
        self.assertTrue(snap["probe_deferred"])


class TestFormatTransparencyRows(unittest.TestCase):
    _SAMPLE_SNAPSHOT = {
        "build": {
            "backend_hint": "vulkan",
            "supports_gpu_offload": True,
            "llama_cpp_python_version": "0.3.29",
        },
        "hardware": {
            "gpu_memory_kind_label": "AMD APU (unified system memory)",
            "vram_budget_gb": 16.0,
            "max_safe_n_gpu_layers": 70,
            "is_unified_gpu_memory": True,
        },
        "settings": {
            "engine_mode": "internal",
            "n_gpu_layers": 33,
            "n_threads": 8,
        },
        "native": {
            "loaded": True,
            "model_basename": "chat.gguf",
            "model_n_params_label": "7.00B",
            "model_n_layers": 32,
            "layer_configuration": "33 of 32 model layers requested (GPU build)",
        },
        "embedder": {"backend": "gpu", "model_basename": "embed.gguf"},
        "sidecar": {"loaded": True, "model_basename": "side.gguf"},
    }

    def test_returns_component_value_pairs(self) -> None:
        rows = format_transparency_rows(self._SAMPLE_SNAPSHOT)
        self.assertGreaterEqual(len(rows), 5)
        labels = [label for label, _value in rows]
        self.assertIn("llama.cpp build", labels)
        self.assertIn("Hardware profile", labels)
        self.assertIn("Native chat", labels)
        self.assertIn("Embeddings", labels)
        self.assertIn("Sidecar", labels)

    def test_lines_derive_from_rows(self) -> None:
        rows = format_transparency_rows(self._SAMPLE_SNAPSHOT)
        lines = format_transparency_lines(self._SAMPLE_SNAPSHOT)
        self.assertEqual(lines, [f"{label}: {value}" for label, value in rows])


class TestFormatTransparencyLines(unittest.TestCase):
    def test_includes_apu_and_sidecar(self) -> None:
        lines = format_transparency_lines(
            TestFormatTransparencyRows._SAMPLE_SNAPSHOT
        )
        joined = "\n".join(lines)
        self.assertIn("vulkan", joined)
        self.assertIn("AMD APU", joined)
        self.assertIn("Embeddings", joined)
        self.assertIn("Sidecar", joined)


if __name__ == "__main__":
    unittest.main()
