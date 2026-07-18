"""Tests for unified GPU memory detection and layer caps."""

from __future__ import annotations

import unittest
from unittest import mock

from core.gpu_layers_cap import (
    _is_amd_apu_unified_memory,
    default_internal_n_gpu_layers_suggested,
    detect_gpu_vram_bytes,
    gpu_memory_kind,
    max_safe_n_gpu_layers,
    reset_gpu_vram_cache_for_tests,
)


class TestAmdApuUnifiedDetection(unittest.TestCase):
    def setUp(self) -> None:
        reset_gpu_vram_cache_for_tests()
    def test_apu_requires_small_carveout_and_gtt(self) -> None:
        four_gb = 4 * 1024 * 1024 * 1024
        fourteen_gb = 14 * 1024 * 1024 * 1024
        self.assertTrue(
            _is_amd_apu_unified_memory(
                carveout_bytes=four_gb,
                gtt_bytes=fourteen_gb,
            )
        )

    def test_discrete_amdgpu_vram_not_treated_as_apu(self) -> None:
        eight_gb = 8 * 1024 * 1024 * 1024
        self.assertFalse(
            _is_amd_apu_unified_memory(
                carveout_bytes=eight_gb,
                gtt_bytes=16 * 1024 * 1024 * 1024,
            )
        )

    def test_small_carveout_without_gtt_not_treated_as_apu(self) -> None:
        four_gb = 4 * 1024 * 1024 * 1024
        self.assertFalse(
            _is_amd_apu_unified_memory(
                carveout_bytes=four_gb,
                gtt_bytes=0,
            )
        )

    @mock.patch("core.gpu_layers_cap.sys.platform", "linux")
    @mock.patch("core.gpu_layers_cap._nvidia_vram_bytes", return_value=0)
    @mock.patch("core.gpu_layers_cap._apple_unified_memory_proxy_bytes", return_value=0)
    @mock.patch("core.gpu_layers_cap._unified_memory_proxy_bytes", return_value=16 * 1024**3)
    @mock.patch(
        "core.gpu_layers_cap._linux_amdgpu_gtt_bytes",
        return_value=14 * 1024**3,
    )
    @mock.patch(
        "core.gpu_layers_cap._linux_amdgpu_carveout_bytes",
        return_value=4 * 1024**3,
    )
    def test_detect_uses_unified_proxy_for_apu(
        self,
        *_mocks: object,
    ) -> None:
        self.assertEqual(detect_gpu_vram_bytes(), 16 * 1024**3)
        self.assertEqual(gpu_memory_kind(), "amd_unified")

    @mock.patch("core.gpu_layers_cap.sys.platform", "linux")
    @mock.patch("core.gpu_layers_cap._nvidia_vram_bytes", return_value=0)
    @mock.patch("core.gpu_layers_cap._apple_unified_memory_proxy_bytes", return_value=0)
    @mock.patch(
        "core.gpu_layers_cap._linux_amdgpu_gtt_bytes",
        return_value=14 * 1024**3,
    )
    @mock.patch(
        "core.gpu_layers_cap._linux_amdgpu_carveout_bytes",
        return_value=8 * 1024**3,
    )
    def test_detect_uses_discrete_amdgpu_vram(
        self,
        *_mocks: object,
    ) -> None:
        self.assertEqual(detect_gpu_vram_bytes(), 8 * 1024**3)
        self.assertEqual(gpu_memory_kind(), "amd_discrete")

    def test_unified_proxy_raises_layer_cap(self) -> None:
        sixteen_gb = 16 * 1024**3
        cap = max_safe_n_gpu_layers(vram_bytes=sixteen_gb)
        self.assertGreaterEqual(cap, 70)
        expected_default = max(0, min(cap, int(round(cap * 0.75))))
        self.assertGreater(expected_default, 12)


if __name__ == "__main__":
    unittest.main()
