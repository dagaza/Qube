# Evidence Map

Traceability from claims/decisions to the concrete files and symbols that back them.
Keep this current so INSPECT-style provenance (P4/P8) exists for the *development*
process too, not just the product. Newest at the bottom.

Format:
```
## <claim / decision>
Evidence: <path:symbol or path:line-range>
Confidence: <high|medium|low>
Notes: <caveats, follow-up>
```

## Capability Plane scaffolding exists
Evidence: `core/integrations/capabilities/protocol.py`, `.../urn.py`, `.../model.py`,
`.../__init__.py`
Confidence: high
Notes: Phase 0 drafts; provider client + MCP adapter still pending.

## MCP proof-of-concept connector exists
Evidence: `core/knowledge/connectors/mcp_connector.py` (one-shot `tools/call` subprocess)
Confidence: high
Notes: Candidate to become an adapter over the shared provider client.

## Phase 0 foundation is complete + tested
Evidence: `core/integrations/capabilities/protocol.py`;
`tests/test_capability_urn.py` (43 tests pass across the 4 capability test files, 2026-07-21).
Confidence: high
Notes: `CapabilityMapper` is provider-agnostic; a mock provider satisfies `CapabilityProvider`
(runtime_checkable). Unknown verbs -> DESTRUCTIVE + `needs_review` (P7).

## Phase 0 diff-review hardening landed
Evidence: `core/integrations/capabilities/mapper.py:CapabilityMappingError` (M1/L1/L2),
`core/integrations/capabilities/model.py:fingerprint_descriptors` (L3);
`tests/test_capability_mapper.py:TestUrnCollision` + `TestNamespaceValidation` + `TestSlugConsistency`.
Confidence: high
Notes: URN collisions disambiguated (`-N` + needs_review); camelCase slug consistent;
un-sluggable namespace raises; fingerprint robust to non-JSON schema.

## Adapter-row hit shape that NormalizedHit targets
Evidence: `core/knowledge/connectors/mcp_connector.py` (delegates; rows from
`NormalizedHit.to_evidence_dict`); `core/knowledge/bundle_builder.py:_generic_row_to_evidence`
consumes `title/snippet/full_text/url/_adapter/retrieval_method/_capability`.
Confidence: high
Notes: `_capability` IS now copied into `EvidenceObject.raw_metadata` on the generic path (KI1
resolved for the canonical bundle path).

## P6 guardrail holds in the capability plane
Evidence: grep for `import mcp` / `provider == "mcp"` / `retrieval_method ==` under
`core/integrations/` returns no matches.
Confidence: high
Notes: Re-run this grep during every Self-Review. Provider client hand-rolls JSON-RPC
(`core/integrations/providers/mcp/jsonrpc.py`) so the `mcp` SDK is never imported.

## First real MCP provider exists and satisfies the contract
Evidence: `core/integrations/providers/mcp/client.py:McpCapabilityProvider`;
`core/integrations/providers/mcp/transport/stdio.py:StdioTransport`;
`tests/test_mcp_provider_client.py` (32 tests: fake transport + real stdio + connector).
Confidence: high
Notes: MCP lifecycle initialize -> notifications/initialized -> tools/list -> tools/call;
invoke routes by `raw_ref`, returns NormalizedHit with cap: provenance; dry_run previews writes.

## McpConnector delegates to the provider (single path)
Evidence: `core/knowledge/connectors/mcp_connector.py:McpConnector` (imports
`core.integrations.providers.mcp.McpCapabilityProvider`; no subprocess/JSON-RPC of its own).
Confidence: high
Notes: Read search runs; write/destructive/needs_review default-denied without a grant (P7).
`_adapter` kept as short id, `_capability` full URN (KI2 resolved on this path).

## Consent + descriptor cache persistence (default-deny, drift-aware)
Evidence: `core/integrations/capabilities/persistence.py:evaluate_access`,
`.../persistence.py:ConsentStore`, `.../persistence.py:integrations_dir`;
`tests/test_capability_persistence.py`.
Confidence: high
Notes: Separate files under `user_data_root()/integrations/<provider>/`; discovery never grants;
fingerprint/tier/needs_review drift forces re-consent (P3/P7).

## cap: provenance reaches the UI (P8 end-to-end, canonical path)
Evidence: `core/knowledge/bundle_builder.py:_generic_row_to_evidence` (raw_metadata capability);
`core/knowledge/ui_adapter.py:evidence_to_ui_source` (source_capability);
`tests/test_capability_bundle_wiring.py`.
Confidence: high
Notes: Closes KI1 on the EvidenceBundle -> bundle_to_ui_sources path. LLMWorker main path
now uses append_turn_evidence_bundle_sources (T13).

## Phase 2 slice 0 — McpConnector consent alignment (on disk)
Evidence: `core/knowledge/connectors/mcp_connector.py:_is_permitted`; `.cursor/starfall/decisions.md`
(2026-07-23 McpConnector consent entry); `tests/test_composer_capability_tokens.py:TestMcpConnectorConsentAlignment`.
Confidence: high
Notes: All tiers through evaluate_access; ephemeral READ when grant is None (configured-source opt-in).

## Phase 2 slice 1 / T14 — cap token spine + invoke gate (on disk)
Evidence: `core/integrations/capability_invoke.py:evaluate_invoke_access`,
`.../capability_invoke.py:invoke_gated_capability`; `core/composer_attachments.py` (cap kind/route);
`workers/llm_worker.py` (CAPABILITY branch); `tests/test_composer_capability_tokens.py` (15 tests).
Confidence: high
Notes: Composer attach≠grant (strict); connector ephemeral READ only; WEB promotion guarded;
KI3 closed Run 003.

## Phase 2 slice 3 / T16 — INSPECT cap steps (on disk)
Evidence: `core/integrations/capability_inspect.py:build_capability_inspect_trace`;
`core/knowledge/observability.py` (capability_steps on RetrievalTrace);
`core/knowledge/retrieval_trace_reader.py` (summary line);
`ui/components/retrieval_inspector.py` (Summary/Explain cap steps);
`workers/llm_worker.py` (CAPABILITY trace + record); `tests/test_capability_inspect_steps.py` (9 tests).
Confidence: high
Notes: Provider-agnostic pure builders; P6 clean. Success-path trace only; denied/cited follow-ups documented. T17–T18 remain.

## Phase 2 slice 2 / T15 — integrations/search + palette (on disk)
Evidence: `core/integrations/search/capability_search.py:search_integrations_capabilities`;
`core/composer_mention_search.py` (integrations section); `ui/components/composer_mention_popup.py`;
`tests/test_integrations_capability_search.py` (12 tests).
Confidence: high
Notes: Fuzzy search over cached descriptors; lock/tier from evaluate_access; P6 clean.

## Phase 2 slice 4 / T17 — preset alias + capabilities field (on disk)
Evidence: `core/knowledge/presets.py:KnowledgePreset.capabilities`;
`core/integrations/preset_capability_alias.py:preset_capability_bundle`,
`.../preset_capability_alias.py:invoke_preset_capability_bundle`;
`core/composer_attachments.py:resolve_attachment_routing` (user: → capability);
`workers/llm_worker.py` (preset bundle CAPABILITY path);
`tests/test_preset_capability_alias.py` (T17, 9 tests).
Confidence: high
Notes: Dual grammar (Option A scoped): `@[cap:…]` canonical; `@[tool:user:…]` alias to cap bundle.
Adapter-only presets unchanged.

## Phase 2 slice 5 / T18 — Sources UI source_capability label (on disk)
Evidence: `core/integrations/capabilities/urn.py:CapabilityURN.display_label`;
`core/knowledge/ui_adapter.py:source_type_label_for_row`,
`.../ui_adapter.py:source_provenance_metadata_parts`;
`ui/components/prestige_dialog.py:_source_type_label` (delegates to ui_adapter);
`tests/test_sources_capability_provenance_ui.py` (T18, 7 tests).
Confidence: high
Notes: CitationSourcesDialog shows humanized cap: label + URN metadata; KI2 cap-shaped
`source_adapter` fallback without provider branching (P6 clean). Phase 2 (#60) complete.

## Phase 3 / T19 — Agent scope boundaries (on disk)
Evidence: `core/integrations/agent_scope.py:AgentScope`, `.../agent_scope.py:build_agent_scope_from_attachments`;
`workers/llm_worker.py` (scope registration + CAPABILITY invoke); `tests/test_agent_scope_egress_phase3.py::TestAgentScope`.
Confidence: high
Notes: P1 enforcement — invoke denied when URN not in turn attachment/preset bundle scope.

## Phase 3 / T20 — Step approval + invoke gate (on disk)
Evidence: `core/integrations/step_approval.py:StepApprovalStore`, `.../step_approval.py:requires_step_approval`;
`core/integrations/capability_invoke.py:invoke_gated_capability`, `.../capability_invoke.py:preview_gated_capability`;
`core/integrations/composer_capability_gate.py`; `ui/views/conversations_view.py` (PrestigeDialog gate);
`tests/test_agent_scope_egress_phase3.py::TestStepApproval`, `...::TestSessionEgressAndInvoke`.
Confidence: high
Notes: WRITE/DESTRUCTIVE/needs_review require per-turn approval; dry_run preview path; egress on deny+allow.

## Phase 3 / T21 — Session egress summary (on disk)
Evidence: `core/integrations/session_egress.py:SessionEgressLedger`;
`core/integrations/egress_summary.py:format_session_egress_summary`;
`ui/components/session_egress_panel.py`; `ui/views/telemetry_view.py:set_active_session_id`;
`core/integrations/capability_inspect.py:build_invoke_step` (server_id, group, raw_tool);
`tests/test_agent_scope_egress_phase3.py` (11 tests). Phase 3 (#61) complete.
Confidence: high
Notes: Distinct from `egress_policy.py` (HTTP SSRF). raw_tool gated on Advanced unlock.

## Q1 resolved — dual grammar
Evidence: `.cursor/starfall/decisions.md` (2026-07-23 Q1 entry); `open-questions.md` Q1 answered
Confidence: high
Notes: Unblocks Phase 2 cap parser without preset migration.

## Provider registry resolves by id without importing a provider (Phase 1 slice 1/3)
Evidence: `core/integrations/registry/provider_registry.py:register_capability_provider`,
`.../provider_registry.py:create_capability_provider`, `.../provider_registry.py:ensure_providers_registered`;
`core/integrations/providers/__init__.py:register_builtin_providers`;
`tests/test_capability_registry.py` (T11, 10 tests).
Confidence: high
Notes: Registry core imports no provider; the composition root is the sole importer of
McpCapabilityProvider. T11 asserts the built-in `mcp` id resolves to a CapabilityProvider (isinstance) and
that the registry source is P6-regex clean. P6 grep clean under core/integrations/.

## LLMWorker main-path cap: provenance (KI1 closed, T13)
Evidence: `workers/llm_worker.py:append_turn_evidence_bundle_sources` path;
`core/knowledge/ui_adapter.py:append_turn_evidence_bundle_sources`;
`tests/test_llmworker_ui_sources.py` (T13).
Confidence: high
Notes: CAPABILITY branch (T14) reuses same bundle→UI path via build_generic_bundle +
append_turn_evidence_bundle_sources.


## Test-plan verifier hazard (Run 001 turn 4)
Evidence: `.cursor/starfall/verify/base.py:_TEST_TOKEN_RE`; `.cursor/starfall/test-plan.md` footnote
Confidence: high
Notes: Prose containing literal `tests/*.py` inside STATUS: COMPLETE test-plan expands to entire suite;
  Phase 2 cases moved to test-plan-phase2.md.
