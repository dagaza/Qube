from __future__ import annotations

import tempfile
import unittest

from core.model_capability_detection import (
    detect_capabilities,
    extract_readme_signals,
    normalize_model_id,
)
from core.model_capability_service import ModelCapabilityService
from core.system_capabilities_store import SystemCapabilitiesStore


class TestModelCapabilitySystem(unittest.TestCase):
    def test_only_heuristics(self) -> None:
        model = {
            "id": "heuristic-model",
            "name": "Agent Tool Runner",
            "description": "reasoning and chain-of-thought enabled",
        }
        caps = detect_capabilities(model)
        self.assertTrue(caps.reasoning.value)
        self.assertTrue(caps.tool_use.value)
        self.assertFalse(caps.vision.value)

    def test_conflicting_signals_highest_confidence_wins(self) -> None:
        model = {
            "id": "conflict-model",
            "name": "vision agent",
            "description": "vision model",
            "files": {"has_mmproj": True, "formats": []},
        }
        caps = detect_capabilities(model, user_overrides={"vision": False})
        self.assertFalse(caps.vision.value)  # override confidence 1.0 wins over file_inference 0.9

    def test_override_wins(self) -> None:
        model = {
            "id": "override-model",
            "name": "plain model",
            "description": "",
            "huggingface": {"pipeline_tag": "image-to-text", "tags": []},
        }
        caps = detect_capabilities(model, user_overrides={"vision": False})
        self.assertFalse(caps.vision.value)
        self.assertAlmostEqual(caps.vision.sources[0].confidence, 1.0)

    def test_vision_via_mmproj(self) -> None:
        model = {
            "id": "mmproj-model",
            "files": {"has_mmproj": True, "formats": []},
        }
        caps = detect_capabilities(model)
        self.assertTrue(caps.vision.value)
        self.assertGreaterEqual(caps.vision.confidence, 0.85)

    def test_hf_only_detection(self) -> None:
        model = {
            "id": "hf-only",
            "huggingface": {
                "pipeline_tag": "automatic-speech-recognition",
                "tags": ["tool", "function-calling"],
            },
        }
        caps = detect_capabilities(model)
        self.assertTrue(caps.stt.value)
        self.assertTrue(caps.tool_use.value)

    def test_pattern_match_across_variants(self) -> None:
        registry = {
            "exact": {},
            "patterns": [
                {
                    "match": "deepseek-r1",
                    "type": "prefix",
                    "capabilities": {"reasoning": True},
                }
            ],
        }
        model = {"id": "TheBloke/deepseek-r1-distill-gguf"}
        caps = detect_capabilities(model, curated_registry=registry)
        self.assertTrue(caps.reasoning.value)

    def test_normalization_improves_detection(self) -> None:
        self.assertEqual(normalize_model_id("TheBloke/deepseek-r1-GGUF"), "deepseek-r1")
        self.assertEqual(normalize_model_id("Qwen2.5-7B-Instruct"), "qwen2.5")

    def test_curated_exact_over_pattern(self) -> None:
        registry = {
            "exact": {"foo/model": {"vision": False}},
            "patterns": [{"match": "model", "type": "contains", "capabilities": {"vision": True}}],
        }
        caps = detect_capabilities({"id": "foo/model"}, curated_registry=registry)
        self.assertFalse(caps.vision.value)

    def test_pattern_over_heuristic(self) -> None:
        registry = {
            "exact": {},
            "patterns": [{"match": "qwen2.5", "type": "prefix", "capabilities": {"tool_use": True}}],
        }
        model = {"id": "qwen2.5-7b", "name": "instruct model"}  # heuristic also positive
        caps = detect_capabilities(model, curated_registry=registry)
        self.assertTrue(caps.tool_use.value)
        self.assertTrue(any(s.source == "curated_pattern" for s in caps.tool_use.sources))

    def test_soft_defaults_only_when_no_stronger_signal(self) -> None:
        no_strong = {"id": "xtts-v2"}
        caps1 = detect_capabilities(no_strong)
        self.assertTrue(caps1.tts.value)
        self.assertTrue(caps1.audio.value)
        self.assertTrue(any(s.source == "soft_default" for s in caps1.tts.sources))

        strong = {"id": "xtts-v2", "huggingface": {"pipeline_tag": "text-to-speech", "tags": []}}
        caps2 = detect_capabilities(strong)
        self.assertTrue(caps2.tts.value)
        # still true, but soft default should not be needed when a stronger signal exists
        self.assertFalse(any(s.source == "soft_default" and s.confidence > 0.5 for s in caps2.tts.sources))

    def test_coding_pattern_detection(self) -> None:
        registry = {
            "exact": {},
            "patterns": [
                {"match": "coder", "type": "contains", "capabilities": {"coding": True}},
                {"match": "code", "type": "contains", "capabilities": {"coding": True}},
            ],
        }
        caps = detect_capabilities({"id": "acme/super-coder-7b"}, curated_registry=registry)
        self.assertTrue(caps.coding.value)

    def test_audio_derived_from_stt(self) -> None:
        caps = detect_capabilities({"id": "asr-model", "files": {"formats": ["whisper"]}})
        self.assertTrue(caps.stt.value)
        self.assertTrue(caps.audio.value)

    def test_ocr_inference_from_gguf_filename_tokens(self) -> None:
        model = {
            "id": "ocr-file-model",
            "files": {
                "gguf_filenames": [
                    "foo.ocr-q4_k_m.gguf",
                    "bar-ocr.q8_0.gguf",
                ]
            },
        }
        caps = detect_capabilities(model)
        self.assertTrue(caps.vision.value)

    def test_image_inference_from_gguf_filename_tokens(self) -> None:
        model = {
            "id": "image-file-model",
            "files": {
                "gguf_filenames": [
                    "foo-image-q4_k_m.gguf",
                    "bar.image-q8_0.gguf",
                    "baz-image.gguf",
                ]
            },
        }
        caps = detect_capabilities(model)
        self.assertTrue(caps.vision.value)

    def test_readme_adds_capability_not_found_elsewhere(self) -> None:
        caps = detect_capabilities(
            {
                "id": "readme-coder",
                "readme": "This model is a coding assistant optimized for programming tasks.",
            }
        )
        self.assertTrue(caps.coding.value)
        self.assertTrue(any(s.source == "readme" for s in caps.coding.sources))

    def test_readme_negation_prevents_false_positive(self) -> None:
        text = "This model is not designed for tool use and does not support vision."
        sigs = extract_readme_signals(text)
        caps = {c for c, _ in sigs}
        self.assertNotIn("tool_use", caps)
        self.assertNotIn("vision", caps)

    def test_runtime_detection_overrides_all(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SystemCapabilitiesStore(system_data_dir=td)
            svc = ModelCapabilityService(store=store)
            svc.on_model_loaded("runtime-model", {"vision": True, "tool_use": False})
            caps = svc.get_or_detect(
                {
                    "id": "runtime-model",
                    "name": "plain",
                    "description": "",
                    "huggingface": {"pipeline_tag": "text-to-speech", "tags": ["tool"]},
                },
                force_refresh=True,
            )
            self.assertTrue(caps.vision.value)
            self.assertFalse(caps.tool_use.value)

    def test_enrichment_updates_cached_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SystemCapabilitiesStore(system_data_dir=td)
            svc = ModelCapabilityService(store=store)
            base = svc.get_or_detect({"id": "enrich-model", "name": "base"})
            self.assertFalse(base.coding.value)
            enriched = svc.enrich_capabilities_after_load(
                "enrich-model",
                {"name": "base", "readme": "code generation and programming tasks"},
            )
            self.assertTrue(enriched.coding.value)

    def test_learned_registry_persists_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s1 = SystemCapabilitiesStore(system_data_dir=td)
            svc1 = ModelCapabilityService(store=s1)
            svc1.on_model_loaded("persist-model", {"reasoning": True, "tool_use": True})

            s2 = SystemCapabilitiesStore(system_data_dir=td)
            svc2 = ModelCapabilityService(store=s2)
            caps = svc2.get_or_detect({"id": "persist-model"}, force_refresh=True)
            self.assertTrue(caps.reasoning.value)
            self.assertTrue(caps.tool_use.value)

    def test_missed_detection_logging(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SystemCapabilitiesStore(system_data_dir=td)
            svc = ModelCapabilityService(store=store)
            svc.get_or_detect({"id": "unknown-model", "name": "unknown"})
            content = store.missed_models_path.read_text(encoding="utf-8")
            self.assertIn("unknown-model", content)

    def test_cache_hit_vs_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SystemCapabilitiesStore(system_data_dir=td)
            svc = ModelCapabilityService(store=store)
            model = {
                "id": "cache-model",
                "name": "first",
                "description": "reasoning model",
            }
            first = svc.get_or_detect(model)
            self.assertTrue(first.reasoning.value)

            # Mutate model to a non-reasoning description; cache should still return old value.
            model2 = {"id": "cache-model", "name": "second", "description": ""}
            cached = svc.get_or_detect(model2, force_refresh=False)
            self.assertTrue(cached.reasoning.value)

            refreshed = svc.get_or_detect(model2, force_refresh=True)
            self.assertFalse(refreshed.reasoning.value)


if __name__ == "__main__":
    unittest.main()
