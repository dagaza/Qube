"""T15 — integrations/search v1 + composer palette Integrations section."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.composer_mention_search import group_search_hits, search_composer_mentions, section_label
from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capabilities.persistence import save_descriptor_cache
from core.integrations.consent_controller import ConsentUIState
from core.knowledge.configured_sources import ConfiguredSource, save_configured_source
from core.integrations.search.capability_search import (
    browse_integrations_capabilities,
    format_capability_subtitle,
    is_capability_locked,
    list_cached_provider_ids,
    search_integrations_capabilities,
)

_SEARCH_SRC = Path(__file__).resolve().parents[1] / "core" / "integrations" / "search"
_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)


def _descriptors(tools):
    group = CapabilityMapper().map_tools("mcp", "github", tools)
    return list(group.capabilities)


_TOOLS = [
    RawTool("search_issues", "Find open GitHub issues", {"type": "object"}),
    RawTool("create_issue", "Open a new issue", {"type": "object"}),
]


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        self._orig_cs = cs.user_data_root
        cs.user_data_root = lambda: self._root  # type: ignore[assignment]

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        cs.user_data_root = self._orig_cs  # type: ignore[assignment]
        self._tmp.cleanup()


def _save_github_mcp_source() -> None:
    save_configured_source(
        ConfiguredSource(
            id="github-mcp",
            label="GitHub MCP",
            connector_type="mcp",
            config={
                "command": ["github-mcp.cmd"],
                "namespace": "github",
                "adapter_id": "github-mcp",
            },
        )
    )


class TestCapabilitySearch(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        self.descs = {d.action: d for d in _descriptors(_TOOLS)}
        _save_github_mcp_source()
        save_descriptor_cache("mcp", list(self.descs.values()))

    def test_list_cached_provider_ids(self):
        self.assertEqual(list_cached_provider_ids(), ["mcp"])

    def test_fuzzy_match_namespace(self):
        hits = search_integrations_capabilities("github")
        self.assertTrue(hits)
        self.assertTrue(any("github" in h.label.lower() for h in hits))

    def test_fuzzy_match_description(self):
        hits = search_integrations_capabilities("open")
        self.assertTrue(any("search" in h.descriptor.action for h in hits))

    def test_tier_in_subtitle(self):
        hits = search_integrations_capabilities("create")
        create = next(h for h in hits if h.descriptor.action == "create-issue")
        self.assertIn("write", create.subtitle.lower())
        self.assertIn("!", create.subtitle)

    def test_locked_when_denied(self):
        hits = search_integrations_capabilities("search")
        read = next(h for h in hits if h.descriptor.action == "search-issues")
        self.assertTrue(read.locked)
        self.assertEqual(read.ui_state, ConsentUIState.DENIED)
        self.assertIn("locked", read.subtitle)

    def test_allowed_not_locked_after_grant(self):
        from core.integrations.consent_controller import IntegrationsConsentController

        read = self.descs["search-issues"]
        IntegrationsConsentController("mcp").grant_capability(read)
        hits = search_integrations_capabilities("search")
        row = next(h for h in hits if h.descriptor.action == "search-issues")
        self.assertFalse(row.locked)
        self.assertEqual(row.ui_state, ConsentUIState.ALLOWED)
        self.assertFalse(is_capability_locked(row.ui_state))

    def test_browse_empty_query_lists_all(self):
        hits = browse_integrations_capabilities("")
        self.assertEqual(len(hits), 2)

    def test_format_subtitle_needs_review(self):
        weird = _descriptors([RawTool("frobnicate_widget", "?", {})])[0]
        subtitle = format_capability_subtitle(
            weird,
            ui_state=ConsentUIState.NEEDS_REVIEW,
        )
        self.assertIn("needs review", subtitle)


class TestComposerIntegrationsSection(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        _save_github_mcp_source()
        save_descriptor_cache("mcp", _descriptors(_TOOLS))

    def test_integrations_section_in_global_search(self):
        hits = search_composer_mentions("github", db=None, store=None)
        integration_hits = [h for h in hits if h.section == "integrations"]
        self.assertTrue(integration_hits)
        self.assertTrue(any("[lock]" in h.label for h in integration_hits))

    def test_section_order_places_integrations_after_tools(self):
        hits = search_composer_mentions("github", db=None, store=None)
        grouped = group_search_hits(hits)
        sections = [h.section for h in grouped]
        if "tools" in sections and "integrations" in sections:
            self.assertLess(sections.index("tools"), sections.index("integrations"))

    def test_section_label(self):
        self.assertEqual(section_label("integrations"), "Integrations")


class TestP6Guardrail(unittest.TestCase):
    def test_search_module_has_no_mcp_leak(self):
        hits: list[str] = []
        for py in _SEARCH_SRC.rglob("*.py"):
            content = py.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if any(p.search(line) for p in _P6_PATTERNS):
                    hits.append(f"{py.name}:{i}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
