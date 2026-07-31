"""MCP / Capability initiative verifier plugin.

Extends the base checks with the initiative-specific P6 guardrail: no MCP import
and no ``provider == "mcp"`` branch anywhere outside ``providers/mcp/``. A leak
is an automatic architecture failure (see ``.cursor/starfall/drift-rules.md``).

Selected when ``active-task.md`` declares ``Verifier: mcp``. To add a new
initiative, drop a sibling ``<name>.py`` here exposing a ``Verifier`` class that
subclasses ``BaseVerifier``, and point ``active-task.md`` at it.
"""
from __future__ import annotations

import re

from base import BaseVerifier, Check, REPO


class Verifier(BaseVerifier):
    name = "mcp"

    GUARDRAIL_ROOT = REPO / "core" / "integrations"
    GUARDRAIL_ALLOW = (REPO / "core" / "integrations" / "providers" / "mcp",)
    GUARDRAIL_PATTERNS = (
        r"\bimport\s+mcp\b",
        r"\bfrom\s+mcp\b",
        r"provider\s*==\s*['\"]mcp['\"]",
    )

    def checks(self) -> list:
        return super().checks() + [self.check_guardrail]

    def check_guardrail(self) -> Check:
        if not self.GUARDRAIL_ROOT.exists():
            rel = self.GUARDRAIL_ROOT.relative_to(REPO)
            return Check("guardrail", "N/A", f"{rel} does not exist")
        patterns = [re.compile(p) for p in self.GUARDRAIL_PATTERNS]
        hits: list[str] = []
        for py in self.GUARDRAIL_ROOT.rglob("*.py"):
            if any(str(py).startswith(str(a)) for a in self.GUARDRAIL_ALLOW):
                continue
            try:
                content = py.read_text(encoding="utf-8")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if any(p.search(line) for p in patterns):
                    hits.append(f"{py.relative_to(REPO)}:{i}")
        if hits:
            return Check("guardrail", "BLOCKED", f"P6 leak: {', '.join(hits)}")
        return Check("guardrail", "PASS", f"no MCP leak under {self.GUARDRAIL_ROOT.relative_to(REPO)}")
