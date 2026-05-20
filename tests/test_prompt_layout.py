"""PR1: prompt_layout resolution — curated, family, overrides; default system_ok."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from core.prompt_layout import (
    DEFAULT_PROMPT_LAYOUT,
    PromptLayoutResolution,
    is_degraded_layout,
    load_curated_prompt_layout_registry,
    resolve_prompt_layout,
)
from core.prompt_layout_store import get_override, set_override


class TestPromptLayout(unittest.TestCase):
    def test_default_is_system_ok(self) -> None:
        res = resolve_prompt_layout(model_id="some-random-model-7b-q4_k_m.gguf")
        self.assertEqual(res.layout, "system_ok")
        self.assertEqual(res.source, "default")
        self.assertFalse(res.degraded)

    def test_settings_override_wins(self) -> None:
        res = resolve_prompt_layout(
            model_id="alpaca-7b.gguf",
            settings_override="system_ok",
        )
        self.assertEqual(res.layout, "system_ok")
        self.assertEqual(res.source, "settings")

        res2 = resolve_prompt_layout(
            model_id="qwen2.5-7b.gguf",
            settings_override="flatten_user",
        )
        self.assertEqual(res2.layout, "flatten_user")
        self.assertEqual(res2.source, "settings")
        self.assertTrue(res2.degraded)

    def test_auto_settings_does_not_block_curated(self) -> None:
        registry = {
            "exact": {},
            "patterns": [
                {"match": "alpaca", "type": "contains", "layout": "flatten_user"},
            ],
        }
        res = resolve_prompt_layout(
            model_id="my-alpaca-7b.gguf",
            settings_override="auto",
            curated_registry=registry,
        )
        self.assertEqual(res.layout, "flatten_user")
        self.assertEqual(res.source, "curated_pattern")

    def test_curated_exact_over_pattern(self) -> None:
        registry = {
            "exact": {"special-model": "system_ok"},
            "patterns": [
                {"match": "special", "type": "contains", "layout": "flatten_user"},
            ],
        }
        res = resolve_prompt_layout(
            model_id="special-model",
            curated_registry=registry,
        )
        self.assertEqual(res.layout, "system_ok")
        self.assertEqual(res.source, "curated")

    def test_family_flatten_vicuna(self) -> None:
        res = resolve_prompt_layout(model_id="TheBloke/vicuna-13b-v1.5-GGUF")
        self.assertEqual(res.layout, "flatten_user")
        self.assertIn(res.source, ("family", "curated_pattern"))

    def test_family_flatten_without_curated_registry(self) -> None:
        res = resolve_prompt_layout(
            model_id="custom-vicuna-13b.gguf",
            curated_registry={"exact": {}, "patterns": []},
        )
        self.assertEqual(res.layout, "flatten_user")
        self.assertEqual(res.source, "family")

    def test_mistral_7b_instruct_v03(self) -> None:
        res = resolve_prompt_layout(
            model_id="Mistral-7B-Instruct-v0.3",
            model_display_name="Mistral-7B-Instruct-v0.3",
        )
        self.assertEqual(res.layout, "flatten_user")
        self.assertTrue(res.degraded)
        self.assertIn(res.source, ("curated", "curated_pattern", "family"))

    def test_family_short_system_tinyllama(self) -> None:
        res = resolve_prompt_layout(model_id="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        self.assertEqual(res.layout, "short_system")
        self.assertTrue(res.degraded)

    def test_store_override_beats_curated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/overrides.json"
            with patch("core.prompt_layout_store.OVERRIDE_PATH", path):
                set_override("test-model.gguf", "system_ok")
                registry = {
                    "exact": {"test-model.gguf": "flatten_user"},
                    "patterns": [],
                }
                res = resolve_prompt_layout(
                    model_id="test-model.gguf",
                    curated_registry=registry,
                )
                self.assertEqual(res.layout, "system_ok")
                self.assertEqual(res.source, "user_override")
                self.assertEqual(get_override("test-model.gguf"), "system_ok")

    def test_is_degraded_layout(self) -> None:
        self.assertFalse(is_degraded_layout("system_ok"))
        self.assertTrue(is_degraded_layout("short_system"))
        self.assertTrue(is_degraded_layout("flatten_user"))

    def test_seed_registry_has_prompt_layout_patterns(self) -> None:
        reg = load_curated_prompt_layout_registry()
        patterns = reg.get("patterns") or []
        matches = {p.get("match") for p in patterns if isinstance(p, dict)}
        self.assertIn("alpaca", matches)
        self.assertIn("vicuna", matches)

    def test_resolution_is_frozen_dataclass(self) -> None:
        res = PromptLayoutResolution(
            layout=DEFAULT_PROMPT_LAYOUT,
            source="default",
            degraded=False,
        )
        self.assertEqual(res.layout, "system_ok")


if __name__ == "__main__":
    unittest.main()
