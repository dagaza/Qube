# Active Task (durable ticket)

The loop's execution state. Starfall reads this first every turn and keeps it current.
This is the single source of "what are we doing right now"; `roadmap.md` is the wider plan.

Initiative:
MCP / Capability Integration (Feature #57)

Current phase:
Documentation / Closing (Run 007 turn 3 complete — turn 4 may emit CLOSING TIME).

Current work item:
Phase 2 / #60 slice 4 / T17 — close Run 007 (T18 deferred).

Mode:
Implementation (closure in progress)

Owner:
Starfall coordinator

Verifier:
mcp

Success criteria (Phase 2 slice 4 / T17 — met on branch):
- KnowledgePreset.capabilities field persists canonical cap: URNs.
- @[tool:user:…] alias resolves to preset cap bundle; adapter-only presets unchanged.
- resolve_attachment_routing + LLMWorker CAPABILITY bundle invoke with INSPECT steps.
- T17 9 tests pass; T14–T16 regression pass; starfall_verify PASS.

Current blockers:
None.

Next decision:
Turn 4 Closing — regenerate next.md baton, PR summary, CLOSING TIME (3 coordinator turns met).

Gate status (Phase 2 slice 4 / T17):
G1 Architecture: PASS | G2 Security: PASS | G3 Product: PASS | G4 Tests: PASS
