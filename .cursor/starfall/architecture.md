# Starfall — Architecture Memory

Condensed, fast-reference view of the capability-integration design. Canonical source:
`docs/mcp_capability_architecture_review.md`. Keep this file short; record *living
deltas* from the canonical doc here as the implementation reveals them.

## Principles (P1-P8) — the review lens
| # | Principle | Review question |
|---|-----------|-----------------|
| P1 | Users choose capabilities; models use capabilities. | Can the model gain a capability the user didn't attach? |
| P2 | Attached intentionally, not exposed indiscriminately. | Is any tool auto-injected on connect? |
| P3 | Permissions understood before granted. | Could a user grant "write" without seeing it's write? |
| P4 | Every invocation inspectable + attributable. | Can I trace an answer to `cap:…` + inputs + outputs? |
| P5 | Capabilities extend Qube; Qube doesn't reorganize around a provider. | Provider-shaped code path added? |
| P6 | Provider-agnostic by construction. | Does a module outside `providers/mcp/` import MCP / branch on provider? |
| P7 | Least privilege by default; escalation explicit. | Does anything default-enable write/destructive? |
| P8 | Provenance never lost. | Does the normalized hit carry its `cap:` origin end-to-end? |

P1-P5 are the philosophy; P6-P8 keep it true.

## Load-bearing constructs
- `CapabilityProvider` protocol — `discover / invoke / health / fingerprint` (4 methods).
- `CapabilityURN` — `cap:<provider>:<namespace>/<action>[@version]`; the spine used by
  composer tokens, presets, grants, INSPECT steps, egress logs, exports.
- Planes — control (registry/config), data (invoke path), observability (health/INSPECT).
- Consent — versioned grant against `cap:…@fingerprint`; drift that escalates tier is
  default-deny until re-reviewed.

## Guardrail
No `import mcp` / `if provider == "mcp"` outside `providers/mcp/`. A leak = Gate 1 fail.

## Living deltas
- (none yet — append dated notes as the code diverges from the canonical doc)
