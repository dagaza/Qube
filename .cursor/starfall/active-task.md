# Active Task (durable ticket)

The loop's execution state. Starfall reads this first every turn and keeps it current.
This is the single source of "what are we doing right now"; `roadmap.md` is the wider plan.

Initiative:
MCP / Capability Integration (Feature #57)

Current phase:
Closing (Run 009 turn 2) — starfall_verify + documentation

Current work item:
Phase 3 / #61 — agent scope + egress summary (T19–T21) complete on disk.

Mode:
Implementation (closure)

Owner:
Starfall coordinator

Verifier:
mcp

Success criteria (Phase 3 — met on branch):
- Agent scope blocks out-of-attachment invokes (P1).
- Write/destructive requires per-message step approval (PrestigeDialog + StepApprovalStore).
- Session egress ledger + Telemetry integrations panel list calls per session.
- T19–T21 11 tests pass; T14–T18 regression pass; starfall_verify PASS.

Current blockers:
None.

Next decision:
Turn 2 Closing — starfall_export, PR summary, CLOSING TIME (after starfall_verify PASS).

Gate status (Phase 3 — turn 2):
G1 Architecture: PASS | G2 Security: PASS | G3 Product: PASS | G4 Tests: PASS
