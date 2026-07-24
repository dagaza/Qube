"""T17 — KnowledgePreset.capabilities + @[tool:user:…] alias resolver (Phase 2 / #60 slice 4)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.composer_attachments import parse_attachments, resolve_attachment_routing
from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import NormalizedHit
from core.integrations.capabilities.persistence import ConsentStore, save_descriptor_cache
from core.integrations.preset_capability_alias import (
    PresetCapabilityBundle,
    build_preset_capability_inspect_trace,
    invoke_preset_capability_bundle,
    preset_capability_bundle,
    resolve_preset_capability_urns,
)
from core.integrations.capability_invoke import CapabilityInvokeResult
from core.integrations.registry.provider_registry import (
    register_capability_provider,
    reset_registry_for_tests,
)
from core.knowledge.presets import KnowledgePreset, save_preset
from core.knowledge.registry import (
    adapter_filter_for_composer_tool,
    resolve_turn_knowledge_service,
)
from core.knowledge.types import SERVICE_PRESET_KNOWLEDGE

_INVOKE_SRC = (
    Path(__file__).resolve().parents[1] / "core" / "integrations" / "preset_capability_alias.py"
)
_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)


def _descriptors(tools):
    return _descriptors_with_namespace("docs", tools)


def _descriptors_with_namespace(namespace: str, tools):
    group = CapabilityMapper().map_tools("fake", namespace, tools)
    return list(group.capabilities)


class _FakeProvider:
    provider_id = "fake"

    def __init__(self, **config):
        self.config = config
        self._descriptors = list(config.get("descriptors") or [])

    async def discover(self):
        return self._descriptors

    async def invoke(self, urn, args, *, ctx):
        return [
            NormalizedHit(
                title=f"Hit from {urn.action}",
                snippet=f"answer for {args.get('query', '')}",
                source_cap=urn,
                url="https://example.test/hit",
            )
        ]

    async def health(self):
        from core.integrations.capabilities import HealthState, HealthStatus

        return HealthStatus(state=HealthState.OK)

    def fingerprint(self):
        return "fake-fp"

    def close(self):
        return None


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]
        import core.knowledge.presets as presets_mod

        self._presets_orig = presets_mod.user_data_root
        presets_mod.user_data_root = lambda: self._root  # type: ignore[assignment]
        reset_registry_for_tests()
        register_capability_provider("fake", _FakeProvider)

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        import core.knowledge.presets as presets_mod

        presets_mod.user_data_root = self._presets_orig  # type: ignore[assignment]
        reset_registry_for_tests()
        self._tmp.cleanup()


class TestPresetCapabilitiesField(_TmpRootTestCase):
    def test_preset_roundtrip_with_capabilities(self):
        from core.knowledge.presets import load_preset

        preset = KnowledgePreset(
            id="github-dev",
            label="GitHub Dev",
            capabilities=[
                "cap:fake:github/search-issues",
                "cap:fake:github/search-code",
            ],
        )
        save_preset(preset)
        loaded = load_preset("github-dev")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded.capabilities), 2)
        self.assertEqual(
            resolve_preset_capability_urns("github-dev"),
            (
                "cap:fake:github/search-issues",
                "cap:fake:github/search-code",
            ),
        )

    def test_invalid_capability_urn_rejected_on_validate(self):
        with self.assertRaises(ValueError):
            KnowledgePreset(
                id="bad-cap",
                label="Bad",
                capabilities=["not-a-urn"],
            ).validate()


class TestPresetAliasResolver(_TmpRootTestCase):
    def setUp(self):
        super().setUp()

        tools = [
            RawTool(name="search_issues", description="Search issues"),
            RawTool(name="search_code", description="Search code"),
        ]
        descriptors = _descriptors_with_namespace("github", tools)
        save_descriptor_cache("fake", descriptors)
        for descriptor in descriptors:
            ConsentStore("fake").grant(descriptor)
        self._descriptors = descriptors

    def test_user_tool_resolves_to_cap_bundle(self):
        save_preset(
            KnowledgePreset(
                id="github-dev",
                label="GitHub Dev",
                capabilities=["cap:fake:github/search-issues"],
            )
        )
        bundle = preset_capability_bundle("user:github-dev")
        self.assertIsInstance(bundle, PresetCapabilityBundle)
        assert bundle is not None
        self.assertEqual(bundle.preset_id, "github-dev")
        self.assertEqual(bundle.urns, ("cap:fake:github/search-issues",))

    def test_adapter_only_preset_has_no_cap_bundle(self):
        save_preset(
            KnowledgePreset(
                id="biology",
                label="Biology",
                adapters=["pubmed"],
            )
        )
        self.assertIsNone(preset_capability_bundle("user:biology"))

    def test_attachment_routing_uses_capability_for_cap_preset(self):
        save_preset(
            KnowledgePreset(
                id="github-dev",
                label="GitHub Dev",
                capabilities=["cap:fake:github/search-issues"],
            )
        )
        _, attachments = parse_attachments(
            "@[tool:user:github-dev] What open bugs mention login?"
        )
        routing = resolve_attachment_routing(attachments)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing["route"], "capability")
        self.assertEqual(routing["strategy"], "attachment_preset_capability")
        self.assertEqual(routing["capability_preset_id"], "github-dev")
        self.assertEqual(
            routing["capability_urns"],
            ["cap:fake:github/search-issues"],
        )

    def test_adapter_only_preset_routing_unchanged(self):
        save_preset(
            KnowledgePreset(
                id="biology",
                label="Biology",
                adapters=["pubmed"],
            )
        )
        _, attachments = parse_attachments("@[tool:user:biology] What is CRISPR?")
        routing = resolve_attachment_routing(attachments)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing["route"], "web")
        self.assertEqual(routing["attachment_tool"], "user:biology")
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="user:biology"),
            SERVICE_PRESET_KNOWLEDGE,
        )
        self.assertEqual(
            adapter_filter_for_composer_tool("user:biology"),
            ("pubmed",),
        )

    def test_invoke_preset_capability_bundle_merges_rows(self):
        save_preset(
            KnowledgePreset(
                id="github-dev",
                label="GitHub Dev",
                capabilities=[
                    "cap:fake:github/search-issues",
                    "cap:fake:github/search-code",
                ],
            )
        )
        bundle_result, per_cap = invoke_preset_capability_bundle(
            "github-dev",
            "login bug",
            provider_factory_kwargs={"descriptors": self._descriptors},
        )
        self.assertTrue(bundle_result.allowed)
        self.assertEqual(len(per_cap), 2)
        self.assertEqual(len(bundle_result.rows), 2)
        caps = {
            str(row.get("_capability") or row.get("source_capability"))
            for row in bundle_result.rows
        }
        self.assertIn("cap:fake:github/search-issues", caps)
        self.assertIn("cap:fake:github/search-code", caps)

    def test_p6_guardrail_clean(self):
        text = _INVOKE_SRC.read_text(encoding="utf-8")
        for pattern in _P6_PATTERNS:
            self.assertIsNone(pattern.search(text), msg=pattern.pattern)


class TestPresetInspectTrace(unittest.TestCase):
    def test_build_preset_capability_inspect_trace_structure(self):
        from core.integrations.capabilities.urn import CapabilityURN

        allowed = CapabilityInvokeResult(
            True,
            "ok",
            rows=({"title": "Hit A", "_capability": "cap:fake:github/search-issues"},),
            urn=CapabilityURN.build("fake", "github", "search-issues"),
        )
        denied = CapabilityInvokeResult(
            False,
            "denied",
            urn=CapabilityURN.build("fake", "github", "search-code"),
        )
        bundle = CapabilityInvokeResult(
            True,
            "ok",
            rows=allowed.rows,
            urn=allowed.urn,
        )
        steps = build_preset_capability_inspect_trace(
            preset_id="github-dev",
            preset_label="GitHub Dev",
            query="login bug",
            per_cap_results=[allowed, denied],
            bundle_result=bundle,
            latency_ms=12.5,
        )
        self.assertGreaterEqual(len(steps), 3)
        self.assertEqual(steps[0]["kind"], "attachment")
        self.assertEqual(steps[0]["urn"], "tool:user:github-dev")
        invoke_steps = [s for s in steps if s.get("kind") == "invoke"]
        self.assertEqual(len(invoke_steps), 2)
        self.assertEqual(invoke_steps[0]["bundle_index"], 1)
        self.assertEqual(invoke_steps[0]["allowed"], True)
        self.assertEqual(invoke_steps[1]["bundle_index"], 2)
        self.assertEqual(invoke_steps[1]["allowed"], False)
        bundle_rank = steps[-1]
        self.assertEqual(bundle_rank["kind"], "ranked")
        self.assertTrue(bundle_rank.get("bundle"))
        self.assertEqual(bundle_rank["kept_count"], 1)
        step_nums = [s["step"] for s in steps]
        self.assertEqual(step_nums, list(range(1, len(steps) + 1)))


if __name__ == "__main__":
    unittest.main()
