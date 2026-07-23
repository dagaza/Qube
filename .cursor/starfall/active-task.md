# Active Task (durable ticket)

The loop's execution state. Starfall reads this first every turn and keeps it current.
This is the single source of "what are we doing right now"; `roadmap.md` is the wider plan.

Initiative:
MCP / Capability Integration (Feature #57)

Current phase:
Idle — Run 003 closed (Phase 2 / #60 slice 0+1 complete).

Current work item:
Next Starfall run: T15 (Integrations palette + integrations/search v1).

Mode:
Idle

Owner:
Starfall coordinator

Verifier:
mcp

Success criteria (Phase 2 slice 0+1 / T14 — met on branch):
- McpConnector consent aligned (ephemeral READ → evaluate_access).
- `@[cap:…]` parse/route + strict invoke gate + LLMWorker CAPABILITY branch.
- T14 tests pass (15); starfall_verify PASS.

Current blockers:
None.

Next decision:
Commit slice 0+1 product files, or arm next run with `next.md` for T15.

Gate status (Phase 2 slice 0+1):
G1 Architecture: PASS | G2 Security: PASS | G3 Product: PASS | G4 Tests: PASS
