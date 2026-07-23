# Architecture Drift Rules

Machine-and-human checkable invariants. The **Self-Review** phase checks the actual diff
against these; any FAIL keeps Gate 1 (architecture) and/or Gate 3 (product) BLOCKED.

FAIL if:
- MCP is imported outside `providers/mcp/`, or any module branches on `provider == "mcp"`
  outside `providers/mcp/`. (P6)
- A new tool/registry is created outside the shared capability registry. (P5)
- The UI exposes raw MCP tools directly instead of attached `cap:` capabilities. (P1/P2)
- Any invocation path bypasses INSPECT provenance. (P4)
- A write/destructive capability is reachable without an explicit, understood grant. (P3/P7)
- A `NormalizedHit` loses its `cap:` origin before the UI/citation. (P8)
- A duplicate subsystem is introduced (e.g. `core/mcp/`) instead of integrating with the
  existing `core/integrations/`, `core/knowledge/connectors/`, Live Sources, or
  EvidenceBundle / INSPECT paths. (Repository Cartographer catch.)

Notes:
- These rules are initiative-specific (MCP/capability). Other initiatives may append their
  own FAIL conditions below their own heading.
