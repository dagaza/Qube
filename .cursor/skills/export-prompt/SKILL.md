---
name: export-prompt
description: Generate a ready-to-paste implementation prompt (Claude Sonnet / Copilot Chat / Cursor) from the Starfall handoff artifact. Use at the end of a Discovery/handoff run.
---

# Export Prompt Skill

Turn the approved handoff into a self-contained implementation prompt another agent can
execute. Source of truth: `.cursor/starfall/handoff.md` (must be `STATUS: READY`) plus
`.cursor/starfall/test-plan.md` and the design doc.

Emit this shape (fill the brackets from the handoff / test-plan):
```
You are implementing Qube Feature #<id> (<initiative>).

Read first:
- docs/mcp_capability_architecture_review.md
- .cursor/starfall/handoff.md
- .cursor/starfall/test-plan.md

Constraints (must hold):
- MCP is a provider only; no raw MCP tools exposed.
- No import of MCP / provider branching outside providers/mcp/ (P6).
- Preserve P1-P8; every result carries cap: provenance (P4/P8).

Implement:
1. <file> — <purpose> — acceptance: <criteria>
2. ...

Before editing: confirm the architecture matches the handoff; ask if anything is unclear.
After editing: run the tests in test-plan.md; do not push to main/master.
```

Keep it copy-paste clean (no Starfall-internal jargon the target agent can't act on).

## Chaining across runs (the baton) — prefer the deterministic exporter
The prompt pack is generated **deterministically**, not by the model. At **Closing**
(and any time the roadmap/handoff changes) run:
```
python .cursor/hooks/starfall_export.py            # regenerate next.md + phase-<n>.md (+ review)
python .cursor/hooks/starfall_export.py --phase next   # print just the next baton
python .cursor/hooks/starfall_export.py --phase 2      # print one phase prompt
python .cursor/hooks/starfall_export.py --phase 2 --review
```
It reads `roadmap.md` (phase table + status), `handoff.md` ("Next slice" + constraints +
STATUS), `known-issues.md` (carried-forward KIs), `open-questions.md` (which Q blocks which
phase), and the git branch, then writes to `.cursor/starfall/prompts/`:
- `next.md` — the immediate baton (handoff "Next slice" if present, else the first
  not-complete phase),
- `phase-<n>.md` — one per remaining roadmap phase,
- `phase-<n>-review.md` — the read-only reviewer counterpart.

Same inputs => identical output, every time (no omissions between phases). Use THIS skill's
LLM rendering only to *enrich* specifics the exporter cannot template (e.g. exact acceptance
criteria drawn from freeform handoff prose) — never as the primary generator.

The pack is the baton for the next agent: pasted into a fresh chat by a human, or launched
as independent headless agents via the Cursor CLI / SDK (see the `sdk` skill). Hooks alone
cannot spawn a new top-level agent — they only re-prompt the current conversation.
