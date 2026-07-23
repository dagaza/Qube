"""T16 — INSPECT capability steps (Phase 2 / #60 slice 3)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.capability_inspect import (
    build_attachment_step,
    build_capability_inspect_trace,
    build_cited_step,
    build_invoke_step,
    build_ranked_step,
    build_returned_step,
    capability_steps_from_trace,
    format_capability_steps_summary_line,
    format_capability_steps_text,
    merge_capability_steps_into_trace,
    serialize_capability_steps,
)
from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace
from core.knowledge.bundle_builder import build_generic_bundle

_INSPECT_SRC = Path(__file__).resolve().parents[1] / "core" / "integrations" / "capability_inspect.py"
_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)


def _descriptor():
    tools = [RawTool(name="search_issues", description="Search GitHub issues")]
    group = CapabilityMapper().map_tools("fake", "GitHub", tools)
    return group.capabilities[0]


class TestCapabilityInspectBuilders(unittest.TestCase):
    def test_attachment_step_carries_urn_and_tier(self):
        step = build_attachment_step(
            urn="cap:fake:github/search-issues",
            label="GitHub — Search issues",
            tier="read",
        )
        self.assertEqual(step["kind"], "attachment")
        self.assertEqual(step["urn"], "cap:fake:github/search-issues")
        self.assertEqual(step["tier"], "read")

    def test_invoke_step_allowed_and_denied(self):
        allowed = build_invoke_step(
            urn="cap:fake:github/search-issues",
            query="crash on export",
            allowed=True,
            action="search_issues",
            latency_ms=12.5,
        )
        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["action"], "search_issues")

        denied = build_invoke_step(
            urn="cap:fake:github/search-issues",
            query="crash on export",
            allowed=False,
            reason="not granted",
        )
        self.assertFalse(denied["allowed"])
        self.assertEqual(denied["reason"], "not granted")

    def test_returned_and_ranked_steps(self):
        returned = build_returned_step(raw_count=18, latency_ms=40.0)
        ranked = build_ranked_step(kept_count=3, rejected_count=15, threshold=0.72)
        self.assertEqual(returned["raw_count"], 18)
        self.assertEqual(ranked["kept_count"], 3)
        self.assertEqual(ranked["threshold"], 0.72)

    def test_cited_step(self):
        step = build_cited_step(
            cited_ids=["4821", "4790"],
            source_capabilities=["cap:fake:github/search-issues"],
        )
        self.assertEqual(step["cited_ids"], ["4821", "4790"])
        self.assertEqual(
            step["source_capabilities"],
            ["cap:fake:github/search-issues"],
        )

    def test_build_capability_inspect_trace_success_chain(self):
        descriptor = _descriptor()
        rows = [
            {
                "title": "Issue 4821",
                "_capability": "cap:fake:github/search-issues",
            }
        ]
        steps = build_capability_inspect_trace(
            urn="cap:fake:github/search-issues",
            query="crash on export",
            allowed=True,
            reason="ok",
            rows=rows,
            bundle_source_count=1,
            rejected_count=0,
            latency_ms=22.0,
            descriptor=descriptor,
            cited_ids=["4821"],
        )
        kinds = [step["kind"] for step in steps]
        self.assertEqual(
            kinds,
            ["attachment", "invoke", "returned", "ranked", "cited"],
        )
        self.assertIn("GitHub", steps[0].get("label", ""))
        self.assertEqual(steps[1]["action"], "search-issues")

    def test_build_capability_inspect_trace_denied_stops_after_invoke(self):
        steps = build_capability_inspect_trace(
            urn="cap:fake:github/search-issues",
            query="crash on export",
            allowed=False,
            reason="not granted",
        )
        self.assertEqual([step["kind"] for step in steps], ["attachment", "invoke"])
        self.assertFalse(steps[1]["allowed"])


class TestCapabilityInspectSerialization(unittest.TestCase):
    def test_serialize_and_merge_into_trace(self):
        steps = build_capability_inspect_trace(
            urn="cap:fake:github/search-issues",
            query="test",
            allowed=True,
            rows=[{"title": "Hit", "_capability": "cap:fake:github/search-issues"}],
            bundle_source_count=1,
        )
        serialized = serialize_capability_steps(steps)
        self.assertEqual(len(serialized), 4)

        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[{"title": "Hit", "_adapter": "github"}],
            rejected_count=0,
            latency_ms=5.0,
            knowledge_service="capability",
            retrieval_strategy="attachment_capability",
        )
        trace = build_retrieval_trace(bundle, capability_steps=steps)
        payload = serialize_retrieval_trace(trace, sources=bundle.sources)
        self.assertIn("capability_steps", payload)
        self.assertEqual(len(payload["capability_steps"]), 4)

        merged = merge_capability_steps_into_trace({"event": "retrieval_trace"}, steps)
        self.assertEqual(len(merged["capability_steps"]), 4)
        self.assertEqual(capability_steps_from_trace(merged)[0]["kind"], "attachment")

    def test_format_capability_steps_text(self):
        steps = build_capability_inspect_trace(
            urn="cap:fake:github/search-issues",
            query="crash on export",
            allowed=True,
            rows=[{"title": "Hit"}],
            bundle_source_count=1,
            descriptor=_descriptor(),
        )
        text = format_capability_steps_text(steps)
        self.assertIn("Capability inspect:", text)
        self.assertIn("Attachment", text)
        self.assertIn("cap:fake:github/search-issues", text)
        self.assertIn("search-issues", text)
        summary = format_capability_steps_summary_line(steps)
        self.assertIn("attachment", summary)
        self.assertIn("invoke", summary)


class TestCapabilityInspectP6Guardrail(unittest.TestCase):
    def test_capability_inspect_module_is_p6_clean(self):
        content = _INSPECT_SRC.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), 1):
            for pat in _P6_PATTERNS:
                self.assertIsNone(
                    pat.search(line),
                    f"capability_inspect trips P6 guardrail pattern {pat.pattern!r} "
                    f"at line {line_no}: {line!r}",
                )


if __name__ == "__main__":
    unittest.main()
