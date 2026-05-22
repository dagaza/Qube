"""Tests for verified-catalog hardware recommendations."""

from __future__ import annotations

import unittest

from core.catalog_hardware_recommendation import (
    CatalogFitLevel,
    build_catalog_recommendation_plan,
    build_tour_model_download_body,
    infer_effective_params_b,
)
from core.hardware_capability_profile import HardwareCapabilityProfile, HardwareTier
from core.qube_verified_models import CatalogEntry


def _entry(catalog_id: str, title: str, description: str = "", gguf_repo: str = "") -> CatalogEntry:
    return CatalogEntry(
        catalog_id=catalog_id,
        title=title,
        description=description,
        publisher="test",
        gguf_repo=gguf_repo or f"test/{catalog_id}",
        gguf_repos=(),
    )


class TestCatalogHardwareRecommendation(unittest.TestCase):
    def test_moe_uses_active_params(self) -> None:
        self.assertEqual(
            infer_effective_params_b(
                title="Gemma 4 26B A4B",
                description="Instruction-tuned Gemma 4 MoE",
            ),
            4.0,
        )

    def test_low_spec_recommends_small_models(self) -> None:
        profile = HardwareCapabilityProfile(
            total_ram_gb=16.0,
            total_vram_gb=4.0,
            cpu_cores=8,
            gpu_name="Test GPU",
            gpu_backend="nvidia",
            tier=HardwareTier.COMPACT,
        )
        entries = [
            _entry("gemma-4-e4b-it", "Gemma 4 4B Instruct", "Ideal for laptops."),
            _entry("phi-4-mini-instruct", "Phi-4 Mini Instruct", "lightweight reasoning model"),
            _entry("llama-3.3-70b-instruct", "Llama 3.3 70B Instruct", "For high-end hardware."),
        ]
        plan = build_catalog_recommendation_plan(entries, profile=profile)
        self.assertGreaterEqual(len(plan.recommended), 2)
        rec_ids = {a.catalog_id for a in plan.recommended}
        self.assertIn("gemma-4-e4b-it", rec_ids)
        self.assertIn("phi-4-mini-instruct", rec_ids)
        self.assertNotIn("llama-3.3-70b-instruct", rec_ids)
        self.assertIn("Gemma 4 4B", plan.banner_text)

    def test_high_end_can_recommend_larger_models(self) -> None:
        profile = HardwareCapabilityProfile(
            total_ram_gb=64.0,
            total_vram_gb=24.0,
            cpu_cores=16,
            gpu_name="RTX 4090",
            gpu_backend="nvidia",
            tier=HardwareTier.PERFORMANCE,
        )
        entries = [
            _entry("gemma-4-e4b-it", "Gemma 4 4B Instruct"),
            _entry("mistral-small-24b-instruct", "Mistral Small 24B Instruct"),
        ]
        plan = build_catalog_recommendation_plan(entries, profile=profile)
        fits = {a.catalog_id: a.fit_level for a in plan.assessments}
        self.assertIn(fits["mistral-small-24b-instruct"], (CatalogFitLevel.EXCELLENT, CatalogFitLevel.GOOD))


    def test_tour_body_lists_recommended_models(self) -> None:
        profile = HardwareCapabilityProfile(
            total_ram_gb=16.0,
            total_vram_gb=4.0,
            cpu_cores=8,
            gpu_name="Test GPU",
            gpu_backend="nvidia",
            tier=HardwareTier.COMPACT,
        )
        entries = [
            _entry("gemma-4-e4b-it", "Gemma 4 4B Instruct", "Ideal for laptops."),
            _entry("phi-4-mini-instruct", "Phi-4 Mini Instruct", "lightweight reasoning model"),
        ]
        body = build_tour_model_download_body(entries, profile=profile)
        self.assertIn("Gemma 4 4B Instruct", body)
        self.assertIn("Phi-4 Mini Instruct", body)
        self.assertIn("For your system", body)


if __name__ == "__main__":
    unittest.main()
