# Implementation Handoff



STATUS: READY

Phase 1 (#59) complete on this branch.
Phase 2 (#60) complete on this branch — slices 0–5 (T14–T18).
Phase 3 (#61) complete on this branch — agent scope + egress summary (T19–T21).
Phase 4 (#62) complete on this branch — hardening / GA readiness (T22–T26).



Feature:

MCP / Capability Integration (Feature #57), Phase 4 / #62 — Hardening / GA readiness:
close KI2/KI4, deferred Phase 2–3 follow-ups (denied-path trace, cited-step wiring,
partial bundle deny UX, router opt-in suggestions), GA checklist (architecture §12 +
engineering checklist CONTRIBUTING note).



Approved architecture:

Provider-agnostic Capability Plane; MCP is one `CapabilityProvider`. Gate 1 PASS.

Canonical design: `docs/mcp_capability_architecture_review.md` (P1-P8, §3-§8, §12 checklist).



Delivered (Phase 4 / #62):

```

Slice 1 / T22 — KI2 close (_adapter short id):

core/integrations/capabilities/model.py

  NormalizedHit.to_evidence_dict uses source_cap.namespace for _adapter (not cap: URN).

core/integrations/capability_invoke.py (unchanged overlay — namespace default)



Slice 2 / T23 — KI4 partial preset bundle deny UX:

core/integrations/preset_capability_alias.py

  format_preset_bundle_deny_summary for partial/full deny per-cap reasons.

workers/llm_worker.py

  Appends deny summary to tool_context on preset bundle partial/full deny.



Slice 3 / T24 — Denied-path INSPECT trace:

core/integrations/capability_trace.py

  CapabilityTraceContext, record_capability_retrieval_trace, build_capability_denial_bundle.

workers/llm_worker.py

  Denied/empty capability invokes persist capability_steps in retrieval trace.



Slice 4 / T25 — Cited-step wiring:

core/integrations/capability_trace.py

  extract_citation_ids_from_text, append_cited_step_to_trace, finalize_capability_cited_trace.

workers/llm_worker.py

  _maybe_finalize_capability_cited_trace after citation renumber.



Slice 5 / T26 — Router opt-in suggestions (default off):

core/app_settings.py + assets/config/settings.schema.json

  qube.integrations.router_suggestions_enabled (default false).

core/integrations/router_capability_suggestions.py

  suggest_integration_capabilities (read-tier, granted only; never auto-invokes).

workers/llm_worker.py

  Adds integration_capability_suggestions to routing decision when opt-in enabled.



GA checklist:

CONTRIBUTING.md — Capability Plane vs internal mcp/ package naming rule.

tests/test_capability_hardening_phase4.py (T22–T26, 11 tests)

```



Prior phases (retained on branch — see Phase 1/2/3 sections in prior handoff revisions).



Not in this run (Phase 4 #62 — deferred):

```

Rename internal mcp/ -> routing/ (tracked debt, non-blocker)

SSE / streamable-http remote MCP transport

Privacy report export UI button (formatter ready)

Live Sources bridge into Capability Plane (architecture §11 Phase 4 expansion)

```



Acceptance criteria (Phase 4 / #62):

[x] KI2 closed — `_adapter` is short id; `_capability` carries full URN (P8).

[x] KI4 closed — preset partial-deny surfaces per-cap reasons in tool_context.

[x] Denied/empty capability invokes persist INSPECT capability_steps.

[x] Cited step appended post-answer when model cites sources.

[x] Router integration suggestions opt-in, default off, suggestions-only (P1/P2).

[x] GA CONTRIBUTING note for mcp/ vs routing/ + §12 P1-P8 preserved.

[x] No `import mcp` / `provider == "mcp"` outside `providers/mcp/` (P6).



Constraints (must hold):

- MCP is a provider only; no raw MCP tools exposed as primary UX.

- No `import mcp` / `provider == "mcp"` outside `providers/mcp/` (P6).

- Preserve P1-P8; every result carries `cap:` provenance (P4/P8).

- Nothing defaults to write/destructive; unknown classification is default-deny + review (P7).



Test requirements:

See `.cursor/starfall/test-plan-phase4.md` (T22–T26 COMPLETE).
Phase 3 regression: test-plan-phase3.md cases remain green.
Phase 2 regression: test-plan-phase2.md cases remain green.



Open questions blocking handoff:

None. Q2 resolved — Phase 4 scope = hardening only (Option A).
