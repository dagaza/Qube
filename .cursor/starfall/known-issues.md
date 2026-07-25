# Starfall — Known Issues

Open defects, risks, and gotchas discovered during loops. Newest at the bottom.

Format:

```
## <id> <short title>  [open|mitigated|closed]
Discovered: <YYYY-MM-DD, phase/turn>
Impact: <what breaks / risk>
Workaround: <if any>
Fix owner / next step: <who/what>
```

## KI1 NormalizedHit `_capability` dropped by EvidenceBundle [closed]

Discovered: 2026-07-21, Phase 0 Code (Repo Cartographer evidence)
Impact: `NormalizedHit.to_evidence_dict()` emits `_capability` (full `cap:` URN), but
`bundle_builder`'s `_row_to_evidence` helpers did not copy it into
`EvidenceObject.raw_metadata`, and the main LLMWorker path stripped `_adapter`/
`retrieval_method` before `all_ui_sources`. So `cap:` provenance was preserved at the
foundation (NormalizedHit) but did NOT reach the INSPECT/Sources UI end-to-end (P8).
Mitigated 2026-07-22 (provider slice): `_generic_row_to_evidence` now copies `_capability`
into `raw_metadata`; `evidence_to_ui_source` emits `source_capability` + `retrieval_method`.
cap: provenance reached the UI on the canonical `bundle_to_ui_sources` path (T10).
Closed 2026-07-23 (Phase 1 / #59): LLMWorker main path now calls
`append_turn_evidence_bundle_sources(all_ui_sources, self._turn_evidence_bundle)` when the
turn bundle has sources; `_apply_sequential_source_ids` renumbers mem/rag/cap rows (T13).
Fix owner / next step: none — closed.

## KI2 `_adapter` overloaded with full URN string [closed]

Discovered: 2026-07-21, Phase 0 Code
Impact: `to_evidence_dict()` sets `_adapter = str(source_cap.base)` (a `cap:` URN), whereas
live adapters use short catalog ids (`pubmed`). Authority/diversity/transparency tables in
`bundle_builder` key off `_adapter` as a catalog id, so a URN there may skew those views.
Mitigated 2026-07-22 (provider slice): on the configured-source path, `McpConnector` overlays
`_adapter` with the short configured id and keeps the full URN in `_capability` (and thus in
`raw_metadata`). So authority/diversity keying stays on a short id (T8 asserts this).
Closed 2026-07-24 (Phase 4 / #62): `NormalizedHit.to_evidence_dict()` now uses
`source_cap.namespace` as `_adapter`; invoke path overlay unchanged. T22 regression tests.
Fix owner / next step: none — closed.

## KI4 Preset bundle partial-deny UX opaque [closed]

Discovered: 2026-07-24, Run 007 turn 2 Self-Review (Security expert)
Impact: When a preset bundles multiple capabilities and some are denied, allowed rows merge
into the turn bundle but per-cap deny reasons are not surfaced in user-facing context — user
may not know part of the preset did not run.
Workaround: INSPECT trace records per-cap invoke steps when trace is persisted; denied caps
visible in Retrieval Inspector on success-path turns only.
Closed 2026-07-24 (Phase 4 / #62): `format_preset_bundle_deny_summary` appended to tool_context
on partial/full preset deny; INSPECT trace already records per-cap steps. T23 tests.
Fix owner / next step: none — closed.

## KI3 Run 002 slice 1 reverted before commit [closed]

Discovered: 2026-07-23, Phase 2 Code (Run 002 turn 3)
Impact: T14 cap spine (`capability_invoke`, `@[cap:…]` tokens, LLMWorker CAPABILITY route,
`test_composer_capability_tokens.py`) was built and verified in Run 002 but reverted before
commit; handoff/verifier drift blocked closure until repo matched claims.
Closed 2026-07-23 (Run 003): slice 0+1 rebuilt on disk; T14 15 tests pass; starfall_verify PASS.
Fix owner / next step: none — closed.
