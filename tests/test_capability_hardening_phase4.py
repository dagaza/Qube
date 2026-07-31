"""Phase 4 / #62 — Hardening & GA readiness (T22–T26)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier, NormalizedHit
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.capability_inspect import build_capability_inspect_trace
from core.integrations.capability_trace import (
    CapabilityTraceContext,
    append_cited_step_to_trace,
    build_capability_denial_bundle,
    extract_citation_ids_from_text,
    finalize_capability_cited_trace,
    record_capability_retrieval_trace,
)
from core.integrations.preset_capability_alias import format_preset_bundle_deny_summary
from core.integrations.capability_invoke import CapabilityInvokeResult
from core.integrations.router_capability_suggestions import (
    format_router_capability_suggestions_line,
    suggest_integration_capabilities,
)

_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)
_PHASE4_SRC = (
    Path(__file__).resolve().parents[1] / "core" / "integrations"
)


class TestKI2AdapterShortId(unittest.TestCase):
    def test_normalized_hit_uses_namespace_as_adapter(self):
        cap = CapabilityURN.parse("cap:mcp:github/search-issues@2")
        row = NormalizedHit(
            title="Hit",
            snippet="Snippet",
            source_cap=cap,
        ).to_evidence_dict()
        self.assertEqual(row["_adapter"], "github")
        self.assertEqual(row["_capability"], "cap:mcp:github/search-issues@2")


class TestKI4PresetPartialDeny(unittest.TestCase):
    def _urn(self, action: str):
        tools = [RawTool(name=action, description=action)]
        group = CapabilityMapper().map_tools("fake", "docs", tools)
        return group.capabilities[0].urn

    def test_partial_deny_summary_lists_denied_caps(self):
        allowed_urn = self._urn("search_docs")
        denied_urn = self._urn("write_docs")
        per_cap = [
            CapabilityInvokeResult(True, "ok", rows=({"title": "Hit"},), urn=allowed_urn),
            CapabilityInvokeResult(False, "not granted", urn=denied_urn),
        ]
        summary = format_preset_bundle_deny_summary(
            per_cap,
            preset_label="My preset",
        )
        self.assertIn("1 of 2", summary)
        self.assertIn(str(denied_urn), summary)
        self.assertIn("not granted", summary)

    def test_full_deny_summary(self):
        denied_urn = self._urn("write_docs")
        per_cap = [CapabilityInvokeResult(False, "step approval required", urn=denied_urn)]
        summary = format_preset_bundle_deny_summary(per_cap, preset_label="X")
        self.assertIn("no capabilities ran", summary)
        self.assertIn("step approval required", summary)

    def test_no_summary_when_all_allowed(self):
        allowed_urn = self._urn("search_docs")
        per_cap = [
            CapabilityInvokeResult(True, "ok", rows=({"title": "Hit"},), urn=allowed_urn)
        ]
        self.assertEqual(format_preset_bundle_deny_summary(per_cap), "")


class TestDeniedPathTrace(unittest.TestCase):
    def test_denial_bundle_and_trace_context(self):
        bundle = build_capability_denial_bundle(
            query_raw="q",
            query_resolved="q",
            latency_ms=3.0,
            stop_reason="capability_denied",
        )
        self.assertEqual(len(bundle.sources), 0)
        steps = build_capability_inspect_trace(
            urn="cap:fake:docs/search",
            query="q",
            allowed=False,
            reason="not granted",
        )
        ctx = CapabilityTraceContext(
            cap_steps=steps,
            query_raw="q",
            query_resolved="q",
            latency_ms=3.0,
            preset_id="",
            cap_urn="cap:fake:docs/search",
            session_id="sess",
            turn_id=1,
            kept_rows=[],
        )
        record_capability_retrieval_trace(ctx, db=None, retrieval_profile="balanced")
        self.assertIsNotNone(ctx.trace)
        self.assertEqual(len(ctx.trace.capability_steps), 2)
        self.assertFalse(ctx.trace.capability_steps[1]["allowed"])


class TestCitedStepWiring(unittest.TestCase):
    def test_extract_citation_ids(self):
        self.assertEqual(
            extract_citation_ids_from_text("See [2] and [1] again [2]."),
            ["2", "1"],
        )

    def test_append_cited_step(self):
        steps = build_capability_inspect_trace(
            urn="cap:fake:docs/search",
            query="q",
            allowed=True,
            rows=[{"title": "Hit", "_capability": "cap:fake:docs/search"}],
            bundle_source_count=1,
        )
        updated = append_cited_step_to_trace(
            steps,
            cited_ids=["1"],
            rows=[{"_capability": "cap:fake:docs/search"}],
        )
        self.assertEqual(updated[-1]["kind"], "cited")
        self.assertEqual(updated[-1]["cited_ids"], ["1"])

    def test_finalize_capability_cited_trace_updates_context(self):
        steps = build_capability_inspect_trace(
            urn="cap:fake:docs/search",
            query="q",
            allowed=True,
            rows=[{"title": "Hit", "_capability": "cap:fake:docs/search"}],
            bundle_source_count=1,
        )
        ctx = CapabilityTraceContext(
            cap_steps=steps,
            query_raw="q",
            query_resolved="q",
            latency_ms=1.0,
            preset_id="",
            cap_urn="cap:fake:docs/search",
            session_id="sess",
            turn_id=2,
            kept_rows=[{"_capability": "cap:fake:docs/search"}],
        )
        record_capability_retrieval_trace(ctx, db=MagicMock(), retrieval_profile="balanced")
        updated = finalize_capability_cited_trace(
            ctx,
            final_text="Answer with [1].",
            all_ui_sources=[{"source_capability": "cap:fake:docs/search"}],
            db=MagicMock(),
            retrieval_profile="balanced",
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated[-1]["kind"], "cited")


class TestRouterSuggestions(unittest.TestCase):
    def test_default_off_returns_empty_without_cache(self):
        self.assertEqual(suggest_integration_capabilities(""), [])

    def test_format_suggestions_line(self):
        line = format_router_capability_suggestions_line(
            [{"label": "GitHub — Search", "urn": "cap:fake:gh/search"}]
        )
        self.assertIn("GitHub", line)


class TestPhase4P6Guardrail(unittest.TestCase):
    def test_new_modules_are_p6_clean(self):
        for rel in (
            "capability_trace.py",
            "router_capability_suggestions.py",
        ):
            content = (_PHASE4_SRC / rel).read_text(encoding="utf-8")
            for line_no, line in enumerate(content.splitlines(), 1):
                for pat in _P6_PATTERNS:
                    self.assertIsNone(
                        pat.search(line),
                        f"{rel} trips P6 at line {line_no}: {line!r}",
                    )


if __name__ == "__main__":
    unittest.main()
