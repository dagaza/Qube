# Implementation Handoff



STATUS: READY

Phase 1 (#59) complete on this branch.
Phase 2 (#60) complete on this branch — slices 0–5 (T14–T18).
Phase 3 (#61) complete on this branch — agent scope + egress summary (T19–T21).



Feature:

MCP / Capability Integration (Feature #57), Phase 3 / #61 — Agent scope + egress summary:
scoped capability boundaries (P1), per-step write/destructive approval, session egress
ledger + Telemetry summary UX, enforcement hooks in invoke gate + LLMWorker.



Approved architecture:

Provider-agnostic Capability Plane; MCP is one `CapabilityProvider`. Gate 1 PASS.

Canonical design: `docs/mcp_capability_architecture_review.md` (P1-P8, §3-§8, §12 checklist).



Delivered (Phase 3 / #61):

```

Slice 1 / T19 — Agent scope model + enforcement:

core/integrations/agent_scope.py

  AgentScope, AgentScopeStore, build_agent_scope_from_attachments; P1 scope check
  before invoke (composer attachments + preset bundle URNs).

workers/llm_worker.py

  Registers agent scope from turn attachments; passes scope into CAPABILITY invoke.



Slice 2 / T20 — Per-step approval + invoke gate extensions:

core/integrations/step_approval.py

  StepApprovalStore; requires_step_approval for WRITE/DESTRUCTIVE/needs_review.

core/integrations/capability_invoke.py

  Scope + step-approval gates; preview_gated_capability (dry_run); egress recording;
  InvokeContext session/turn attribution.

core/integrations/composer_capability_gate.py

  Qt-free pending approval discovery + message formatting for PrestigeDialog.

ui/views/conversations_view.py

  Write/destructive cap send gate → PrestigeDialog → step_approval_store.grant_many.



Slice 3 / T21 — Session egress summary:

core/integrations/session_egress.py

  IntegrationEgressRecord, SessionEgressLedger, build_egress_record.

core/integrations/egress_summary.py

  format_session_egress_summary, format_privacy_report_integrations_section;
  raw_tool when Advanced engine unlocked.

core/integrations/capability_inspect.py

  invoke steps carry server_id, capability_group, raw_tool (INSPECT parity).

ui/components/session_egress_panel.py

  Read-only Telemetry card for session integration calls.

ui/views/telemetry_view.py

  Session integrations panel + set_active_session_id refresh hook.

tests/test_agent_scope_egress_phase3.py (T19–T21, 11 tests)

```



Prior phases (retained on branch — see Phase 1/2 sections in prior handoff revisions).



Not in this run (Phase 3 #61 — deferred):

```

Router opt-in capability suggestions (default off — roadmap only)

SSE / streamable-http remote MCP transport (Phase 3 P2 — document BYO)

Full multi-step agent plan UI (scoped single-turn composer path delivered)

One-click privacy report export button (formatter ready; UI export action deferred)

```



Acceptance criteria (Phase 3 / #61):

[x] Agent scope enforces attached/bundled caps only (T19, P1).

[x] Write/destructive requires explicit per-message step approval beyond Settings grant (T20, P3/P7).

[x] dry_run preview path wired (preview_gated_capability; provider honors ctx.dry_run).

[x] Session egress ledger records integration calls (server id, group, tier, allow/deny).

[x] Telemetry shows session integrations summary (Theme B).

[x] INSPECT invoke steps include server_id, capability_group, raw_tool.

[x] No `import mcp` / `provider == "mcp"` outside `providers/mcp/` (P6).



Constraints (must hold):

- MCP is a provider only; no raw MCP tools exposed as primary UX.

- No `import mcp` / `provider == "mcp"` outside `providers/mcp/` (P6).

- Preserve P1-P8; every result carries `cap:` provenance (P4/P8).

- Nothing defaults to write/destructive; unknown classification is default-deny + review (P7).



Test requirements:

See `.cursor/starfall/test-plan-phase3.md` (T19–T21 COMPLETE).
Phase 2 regression: test-plan-phase2.md cases remain green.



Open questions blocking handoff:

None. KI4 partial preset bundle deny UX remains deferred (non-blocker).

Router suggestions + remote transport are Phase 3 roadmap deferrals, not blockers.


