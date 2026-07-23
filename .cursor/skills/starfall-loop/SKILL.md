---
name: starfall-loop
description: Run the architecture-aware Starfall engineering coordinator - a persistent, multi-pass loop with durable execution state, review gates, drift detection, and parallel specialist passes. Use when the user asks to "run starfall", loop, or drive a Qube feature/initiative from discovery to PR-ready handoff.
---

# Starfall Loop

**Read `.cursor/agents/starfall.md` for the full instructions.** This skill is a
quick reference only. The loop is the product; each run drives one **initiative**
(current initiative in `.cursor/starfall/active-task.md`).

## Quick start
1. Read `.cursor/agents/starfall.md` first (source of truth).
2. Armed pre-submit when the prompt contains the keyword "starfall" (`beforeSubmitPrompt`).
3. Each turn: read active-task + memory + log + context, spawn the relevant specialists in
   parallel, advance the phase (honoring gates), append ONE work entry (with the P1-P8
   Architecture Review block), refresh context + memory.
4. Close only when the closure contract holds (see below), then add `CLOSING TIME`.

## Workflow phases
`Discovery -> Architecture Review -> Implementation Plan -> Implementation Handoff ->
Code -> Self-Review -> Testing -> Documentation -> Closing`.
Discovery mode stops at Implementation Handoff (read-only, produces `handoff.md`).

## Review gates + closure contract
- Gates: G1 Architecture · G2 Security · G3 Product (P1-P8 + drift-rules) · G4 Tests.
- Code blocked until G1-G3 PASS; closing blocked until G4 PASS.
- **Closure contract (enforced by the stop hook):** 3+ turns AND structural markers (latest
  `Gates:` line has G1-G3 PASS and G4 PASS-or-`N/A`; `handoff.md` marked `STATUS: READY`)
  AND **executable evidence** — `starfall_verify.py` reports PASS (tests actually run,
  delivered files exist, P6 guardrail clean, git sane). Markers are claims; the verifier
  supplies facts; closure needs both.

## Specialist experts (read-only, parallel)
Repository Cartographer · Capability Architecture · MCP Protocol · Security & Permissions ·
Product Review · Quality (+ UI/Composer, Infra/Config). Skills: `research`, `mcp-provider`,
`export-prompt`, `git`.

## Key files
- `.cursor/agents/starfall.md` — full instructions (source of truth)
- `.cursor/starfall-log.md` — continuous append-only work log; runs are delimited by
  `# Run NNN` section headers (opened by `starfall_prep.py` on a fresh arm), NOT archived each run
- `.cursor/starfall-context.md` — rolling context summary (also gains a `## Run NNN` header per run)
- `.cursor/starfall/settings.json` — hook runtime settings (`diagnostics` on/off; env override `STARFALL_DIAGNOSTICS`)
- `.cursor/hooks/common.py` — shared hook runtime (BOM-tolerant `read_payload`, gated `write_debug`, run mgmt)
- `.cursor/hooks/test_hooks.py` — hook runtime self-test / doctor (`python .cursor/hooks/test_hooks.py`)
- `.cursor/starfall/` — memory + execution state (NOT archived by the hook):
  `active-task.md`, `architecture.md`, `decisions.md`, `known-issues.md`, `roadmap.md`,
  `open-questions.md`, `evidence-map.md`, `test-plan.md`, `handoff.md`, `drift-rules.md`
- `.cursor/starfall/verify/` — verification plugins: `base.py` (agnostic checks) +
  `<initiative>.py` (e.g. `mcp.py`); selected by the `Verifier:` field in `active-task.md`
- `.cursor/starfall/prompts/` — deterministically generated batons: `next.md`,
  `phase-<n>.md`, `phase-<n>-review.md` (regenerate via the exporter; do not hand-edit)
- `.cursor/starfall/subagents.log` — durable ledger of Task-subagent completions (gitignored)
- `.cursor/.starfall-mode` / `.cursor/.starfall-lock` — trigger / parallel-loop guard
- `.cursor/starfall-archive/` — rolled-over log/context (only on size >512 KB, age >30 days,
  or explicit `.cursor/.starfall-archive-now` sentinel), not per-run

## Hooks (Python, cross-platform)
- `beforeSubmitPrompt` -> `.cursor/hooks/starfall_prep.py` — arms the trigger
- `stop` -> `.cursor/hooks/starfall.py` — continues the loop; enforces the closure contract
- `stop` (evidence) -> `.cursor/hooks/starfall_verify.py` — orchestrator called by the stop
  hook; turns declared closure state into verified state. Loads the verifier plugin named in
  `active-task.md` (`Verifier:` field): `.cursor/starfall/verify/base.py` (tests, files, work
  log, evidence map, git) + an initiative plugin (`verify/mcp.py` adds the P6 guardrail).
  Fail-safe: missing evidence behind a claim BLOCKS. Run by hand: `python
  .cursor/hooks/starfall_verify.py` (exit 0 = PASS).
- `beforeShellExecution` -> `.cursor/hooks/block_main_commit.py` — blocks commit/push to
  main/master and local merges
- `beforeShellExecution` -> `.cursor/hooks/verify_commit.py` — runs the verifier before a
  `git commit` while a run is armed; denies the commit if the repo can't back the claims.
- `subagentStop` -> `.cursor/hooks/starfall_subagent.py` — appends each Task-subagent
  completion to `.cursor/starfall/subagents.log` (external proof of multi-agent activity).

## Deterministic exporter (the baton generator)
`python .cursor/hooks/starfall_export.py` regenerates `.cursor/starfall/prompts/` from
`roadmap.md` + `handoff.md` + `known-issues.md` + `open-questions.md` + git branch — no LLM,
identical output every run. `--phase next|<n>` prints one; `--phase <n> --review` prints the
reviewer prompt. Run it at Closing so the next agent's baton is always current.

## Log format
Avoid leading dashes in work entries (hooks reserve dash/`status:` lines for metadata):
```markdown
## <Expert(s)> - <timestamp>
Phase: <current phase>
Gates: G1 <PASS|BLOCKED:…> | G2 <…> | G3 <…> | G4 <…>
Signals / Actions / Decisions / Next step: ...

Architecture Review
[ ] P1 ... [ ] P8   (tick each, or annotate N/A with reason)
```
