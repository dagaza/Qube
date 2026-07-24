# Active Task (durable ticket)

The loop's execution state. Starfall reads this first every turn and keeps it current.
This is the single source of "what are we doing right now"; `roadmap.md` is the wider plan.

Initiative:
MCP / Capability Integration (Feature #57)

Current phase:
Testing complete (Run 008 turn 3) — next: Documentation / Closing (turn 4)

Current work item:
Phase 2 / #60 slice 5 / T18 — close Run 008 (Phase 2 complete).

Mode:
Implementation (closure)

Owner:
Starfall coordinator

Verifier:
mcp

Success criteria (Phase 2 slice 5 / T18 — met on branch):
- CitationSourcesDialog type label prefers `source_capability` URN display over adapter-only labels.
- KI2 overlay: cap-shaped `source_adapter` falls back to URN label when feasible.
- Non-cap rows unchanged (adapter / type fallbacks).
- T18 7 tests pass; T14–T17 + T13 regression pass (56 tests); starfall_verify PASS.

Current blockers:
None.

Next decision:
Turn 4 Documentation/Closing — starfall_export next.md baton, PR summary, CLOSING TIME (3 coordinator turns now met).

Gate status (Phase 2 slice 5 / T18 — turn 3):
G1 Architecture: PASS | G2 Security: PASS | G3 Product: PASS | G4 Tests: PASS
