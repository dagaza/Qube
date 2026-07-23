# Active Task (durable ticket)

The loop's execution state. Starfall reads this first every turn and keeps it current.
This is the single source of "what are we doing right now"; `roadmap.md` is the wider plan.

Initiative:
MCP / Capability Integration (Feature #57)

Current phase:
Idle — Run 004 closed (Phase 2 / #60 slice 2 / T15 complete).

Current work item:
Next Starfall run: T16 (INSPECT cap steps).

Mode:
Idle

Owner:
Starfall coordinator

Verifier:
mcp

Success criteria (Phase 2 slice 2 / T15 — met on branch):
- `core/integrations/search/` fuzzy capability search over cached descriptors.
- Composer global search Integrations section with tier/lock hints.
- Popup selects `@[cap:…]` attachments; T15 12 tests pass; starfall_verify PASS.

Current blockers:
None.

Next decision:
Commit slice 2 product files, or arm next run with `next.md` for T16.

Gate status (Phase 2 slice 2 / T15):
G1 Architecture: PASS | G2 Security: PASS | G3 Product: PASS | G4 Tests: PASS
