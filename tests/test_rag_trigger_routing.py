"""Tests for custom NLP RAG trigger routing (Settings -> LLMWorker)."""

from __future__ import annotations

import os
import re
import unittest

from core.rag_trigger_routing import (
    apply_custom_rag_trigger_route,
    matches_custom_rag_trigger,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MatchesCustomRagTriggerTests(unittest.TestCase):
    def test_empty_inputs(self) -> None:
        self.assertFalse(matches_custom_rag_trigger("", ["search my files"]))
        self.assertFalse(matches_custom_rag_trigger("search my files", []))

    def test_case_insensitive_prompt_matching(self) -> None:
        triggers = ("search my files",)
        self.assertTrue(
            matches_custom_rag_trigger(
                "please search my files for the budget",
                triggers,
            )
        )

    def test_no_match(self) -> None:
        self.assertFalse(
            matches_custom_rag_trigger(
                "what is the weather today",
                ("search my files",),
            )
        )


class ApplyCustomRagTriggerRouteTests(unittest.TestCase):
    def test_no_match_is_noop(self) -> None:
        route, force = apply_custom_rag_trigger_route("HYBRID", matched=False)
        self.assertEqual(route, "HYBRID")
        self.assertFalse(force)

    def test_none_upgrades_to_rag_with_bypass(self) -> None:
        route, force = apply_custom_rag_trigger_route("NONE", matched=True)
        self.assertEqual(route, "RAG")
        self.assertTrue(force)

    def test_hybrid_preserved_with_bypass(self) -> None:
        route, force = apply_custom_rag_trigger_route("HYBRID", matched=True)
        self.assertEqual(route, "HYBRID")
        self.assertTrue(force)

    def test_memory_upgrades_to_hybrid(self) -> None:
        route, force = apply_custom_rag_trigger_route("MEMORY", matched=True)
        self.assertEqual(route, "HYBRID")
        self.assertTrue(force)

    def test_rag_stays_rag_with_bypass(self) -> None:
        route, force = apply_custom_rag_trigger_route("RAG", matched=True)
        self.assertEqual(route, "RAG")
        self.assertTrue(force)

    def test_web_unchanged_without_bypass(self) -> None:
        route, force = apply_custom_rag_trigger_route("WEB", matched=True)
        self.assertEqual(route, "WEB")
        self.assertFalse(force)


class LLMWorkerRagTriggerContractTests(unittest.TestCase):
    """Static contract checks on ``workers/llm_worker.py`` wiring."""

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "workers", "llm_worker.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_uses_rag_trigger_routing_helpers(self) -> None:
        self.assertIn("apply_custom_rag_trigger_route", self.src)
        self.assertIn("matches_custom_rag_trigger", self.src)
        self.assertIn("from core.rag_trigger_routing import", self.src)

    def test_refresh_rag_triggers_method_exists(self) -> None:
        self.assertRegex(self.src, r"def refresh_rag_triggers\s*\(")

    def test_master_off_bypass_on_rag_search(self) -> None:
        self.assertRegex(
            self.src,
            r'execution_route in \["RAG", "HYBRID"\] and \(\s*'
            r"self\.mcp_rag_enabled or force_rag_via_trigger",
        )

    def test_does_not_blindly_assign_rag_on_trigger(self) -> None:
        self.assertNotRegex(
            self.src,
            re.compile(
                r"matches_custom_rag_trigger[\s\S]{0,200}?"
                r'execution_route\s*=\s*"RAG"',
                re.MULTILINE,
            ),
        )


class SettingsViewRagTriggerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths = [
            os.path.join(ROOT, "ui", "views", "settings", "settings_view.py"),
            os.path.join(ROOT, "ui", "views", "settings_view.py"),
        ]
        handlers_dir = os.path.join(ROOT, "ui", "views", "settings", "handlers")
        if os.path.isdir(handlers_dir):
            for name in sorted(os.listdir(handlers_dir)):
                if name.endswith(".py"):
                    paths.append(os.path.join(handlers_dir, name))
        cls.src = ""
        for path in paths:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    cls.src += f.read()

    def test_settings_refreshes_worker_cache_on_change(self) -> None:
        self.assertIn("refresh_rag_triggers", self.src)
        self.assertIn("_refresh_llm_rag_triggers", self.src)


if __name__ == "__main__":
    unittest.main()
