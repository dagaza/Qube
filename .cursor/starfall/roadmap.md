# Starfall — Roadmap & Phase Status

Phase -> Azure DevOps work item -> status. Source: `docs/mcp_capability_integrations_plan.md`
(§ Phases) and Feature #57. Update the Status column each loop.

| Phase | Scope | Work item | Status |
|-------|-------|-----------|--------|
| 0 | Foundations — `CapabilityProvider`, `CapabilityURN`, model, mapper, first MCP provider | #58 | COMPLETE — foundation (T1-T7) + first real `McpCapabilityProvider` (persistent stdio JSON-RPC), `McpConnector` delegation, consent/descriptor persistence, and KI1 bundle→UI wiring. 75 capability/provider tests (T1-T10) pass. Phase 1 (UI + permission model) is next. |
| 1 | Integrations UI + permission model | #59 | COMPLETE — provider registry (T11), Qt-free IntegrationsConsentController + Settings → Integrations section (T12), LLMWorker main-path `append_turn_evidence_bundle_sources` / bundle_to_ui_sources migration (T13, KI1 closed). 97 capability/integration tests pass. |
| 2 | Composer palette + presets + INSPECT | #60 | Slice 0+1 COMPLETE (T14); slice 2 COMPLETE (T15); slice 3 COMPLETE (T16); slices 4–5 (T17–T18) pending. |
| 3 | Agent scope + egress summary | #61 | Not started |
| 4 | Hardening / GA readiness (roadmap §11) | #62 | Not started |

## Notes
- Phase numbering follows the plan doc; #62/Phase 4 tracks hardening/GA beyond the
  four plan phases. Confirm scope with the parent Feature #57 as it firms up.
