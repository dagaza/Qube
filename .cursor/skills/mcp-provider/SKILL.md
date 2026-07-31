---
name: mcp-provider
description: MCP implementation rules for Qube capability providers. Use when designing or reviewing anything under providers/mcp/ or the MCP transport/lifecycle.
---

# MCP Provider Rules

MCP is a **provider**, not the architecture. It implements the `CapabilityProvider`
contract behind the provider-agnostic Capability Plane.

Never:
- expose raw MCP tools directly to the model or UI
- `import mcp` / branch on `provider == "mcp"` outside `providers/mcp/`
- bypass `CapabilityURN` (everything is addressed as `cap:mcp:<ns>/<action>[@version]`)
- bypass permission evaluation or INSPECT provenance

Required lifecycle:
```
initialize
  -> capability discovery      (tools/list -> CapabilityDescriptor[])
  -> descriptor mapping        (to CapabilityURN + tier + input schema)
  -> permission evaluation     (grant vs default-deny; drift re-consent)
  -> invoke                    (tools/call)
  -> NormalizedHit             (carries source_cap = cap:mcp:…)
  -> INSPECT provenance        (cap: -> inputs -> outputs -> citation)
```

Transport: support stdio and HTTP; keep JSON-RPC lifecycle + protocol compliance inside
`providers/mcp/`. Health flows through the shared Source Status surface, not a bespoke
widget. See `docs/mcp_capability_architecture_review.md` §§3-6.
