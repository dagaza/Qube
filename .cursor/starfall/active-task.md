# Active Task (durable ticket)

The loop's execution state. Starfall reads this first every turn and keeps it current.
This is the single source of "what are we doing right now"; `roadmap.md` is the wider plan.

Initiative:
MCP / Capability Integration (Feature #57)

Current phase:
Idle — Run 006 closed (Phase 2 / #60 slice 3 / T16 complete).

Current work item:
Next Starfall run: T17 (KnowledgePreset.capabilities + @[tool:user:…] alias resolver).

Mode:
Idle

Owner:
Starfall coordinator

Verifier:
mcp

Success criteria (Phase 2 slice 3 / T16 — met on branch):
- Pure capability INSPECT step builders (attachment→invoke→returned→ranked→cited).
- capability_steps serialized on retrieval trace; Retrieval Inspector Summary/Explain render them.
- LLMWorker CAPABILITY route records trace + RetrievalRecord with steps (success path).
- T16 9 tests pass; T14+T15 regression pass; starfall_verify PASS.

Current blockers:
None.

Next decision:
Commit slice 3 product files, or arm next run with `next.md` for T17.

Gate status (Phase 2 slice 3 / T16):
G1 Architecture: PASS | G2 Security: PASS | G3 Product: PASS | G4 Tests: PASS
