# Implementation Handoff



STATUS: READY

Phase 1 (#59) complete on this branch (see Delivered section below).
Phase 2 (#60) slice 0+1 (T14) + slice 2 (T15) + slice 3 (T16) + slice 4 (T17) delivered on disk. Next run: T18 Sources UI label.



Feature:

MCP / Capability Integration (Feature #57), Phase 1 / #59 — Integrations UI + permission

model: (1) provider registry, (2) permission/consent UI, (3) LLMWorker main-path cap:

provenance migration (KI1 closed). Prior slice (#58 continuation) is DELIVERED on this branch.



Approved architecture:

Provider-agnostic Capability Plane; MCP is one `CapabilityProvider`. Gate 1 PASS.

Canonical design: `docs/mcp_capability_architecture_review.md` (P1-P8, §3-§8, §12 checklist).



Delivered (Phase 1 / #59 — all three slices):

```

Slice 1/3 — Provider registry:

core/integrations/registry/__init__.py

core/integrations/registry/provider_registry.py

core/integrations/providers/__init__.py (composition root)

tests/test_capability_registry.py (T11)



Slice 2/3 — Permission/consent UI:

core/integrations/consent_controller.py

  Qt-free IntegrationsConsentController — groups by CapabilityDescriptor.group; exposes tier +

  needs_review; per-capability state from evaluate_access (not grant presence); grant()/deny()

  write ConsentStore on the exact descriptor; needs_review un-grantable; drift/tier-escalation

  surfaces REREVIEW_REQUIRED (P3/P7).

ui/views/settings/sections/integrations.py

  Thin PyQt6 Settings → Integrations section (PrestigeToggle per capability, tier badges).

ui/views/settings/registry.py (+ section builders wired)

tests/test_integrations_consent_controller.py (T12)



Slice 3/3 — LLMWorker main-path UI (KI1 close):

core/knowledge/ui_adapter.py — append_turn_evidence_bundle_sources()

workers/llm_worker.py — when _turn_evidence_bundle has sources, append via bundle_to_ui_sources

  instead of the manual web_results loop; _apply_sequential_source_ids renumbers mem/rag/web/cap ids.

tests/test_llmworker_ui_sources.py (T13)

```



Prior slice (#58, delivered — retained for provenance):

```

core/integrations/providers/mcp/__init__.py

core/integrations/providers/mcp/client.py

core/integrations/providers/mcp/jsonrpc.py

core/integrations/providers/mcp/transport/__init__.py

core/integrations/providers/mcp/transport/base.py

core/integrations/providers/mcp/transport/stdio.py

core/integrations/capabilities/persistence.py

core/integrations/capabilities/__init__.py

core/knowledge/connectors/mcp_connector.py

core/knowledge/bundle_builder.py

core/knowledge/ui_adapter.py

tests/fixtures/mock_mcp_server.py

tests/test_mcp_provider_client.py

tests/test_capability_persistence.py

tests/test_capability_bundle_wiring.py

```



Delivered (Phase 2 / #60 — slice 0+1):

```

Slice 0 — consent alignment (G2):

core/knowledge/connectors/mcp_connector.py

  _is_permitted: remove READ bypass; ephemeral PermissionGrant when grant is None
  (configured-source read opt-in) → evaluate_access for all tiers.

Slice 1 / T14 — cap token spine + invoke gate:

core/integrations/capability_invoke.py

  evaluate_invoke_access (strict; attach ≠ grant) + invoke_gated_capability.

core/composer_attachments.py

  AttachmentKind capability; @[cap:…] parse (fail-closed); route:capability patch.

workers/llm_worker.py

  CAPABILITY route → invoke_gated_capability → build_generic_bundle +
  append_turn_evidence_bundle_sources; WEB promotion guarded.

tests/test_composer_capability_tokens.py (T14)

```



Partial (Phase 2 / #60 — prior on disk):

```

core/knowledge/connectors/mcp_connector.py

  Delegates to McpCapabilityProvider (persistent session).

```



Delivered (Phase 2 / #60 — slice 2 / T15):

```

Slice 2 / T15 — integrations/search v1 + Integrations palette section:

core/integrations/search/__init__.py

core/integrations/search/capability_search.py

  list_cached_provider_ids, fuzzy search, tier/lock from evaluate_access;
  CapabilityPaletteEntry for composer rows.

core/composer_mention_search.py

  "integrations" section in global search (after tools); @[cap:…] attach payload.

ui/components/composer_mention_popup.py

  CapabilityPaletteEntry tooltips + ComposerAttachment(kind=capability) on select.

tests/test_integrations_capability_search.py (T15, 12 tests)

```



Delivered (Phase 2 / #60 — slice 3 / T16):

```

Slice 3 / T16 — INSPECT cap steps:

core/integrations/capability_inspect.py

  Pure builders (attachment→invoke→returned→ranked→cited); trace merge + text formatters.

core/knowledge/observability.py

  RetrievalTrace.capability_steps + serialize into JSONL trace payload.

core/knowledge/retrieval_trace_reader.py

  Summary line when capability_steps present.

ui/components/retrieval_inspector.py

  Summary + Explain tabs render capability INSPECT steps from trace.

workers/llm_worker.py

  CAPABILITY route builds steps, records retrieval trace + RetrievalRecord.

tests/test_capability_inspect_steps.py (T16, 9 tests)

```



Delivered (Phase 2 / #60 — slice 4 / T17):

```

Slice 4 / T17 — KnowledgePreset.capabilities + @[tool:user:…] alias resolver:

core/knowledge/presets.py

  KnowledgePreset.capabilities field (canonical cap: URNs); validate/dedupe; adapter-only presets unchanged.

core/integrations/preset_capability_alias.py

  preset_capability_bundle resolver; invoke_preset_capability_bundle; INSPECT trace for alias→bundle.

core/composer_attachments.py

  resolve_attachment_routing: user: preset with capabilities → route:capability (dual grammar).

workers/llm_worker.py

  CAPABILITY route invokes preset bundles; preset_id on retrieval fingerprint.

tests/test_preset_capability_alias.py (T17, 9 tests)

```



Not in this run (Phase 2 #60 — slice 5):

```

T18 — Sources UI source_capability label

```



Acceptance criteria (Phase 1 / #59):

[x] Provider registry resolves by string id without importing concrete providers (T11, P5/P6).

[x] Consent controller lists groups + tiers + needs_review; state from evaluate_access; grant/deny

    persist exact descriptor; needs_review un-grantable; drift/tier-escalation → re-review (T12, P3/P7).

[x] Settings → Integrations section renders controller rows (no raw MCP tools as primary UX).

[x] LLMWorker main path uses bundle_to_ui_sources when turn EvidenceBundle present; citation ids

    unique after renumber; cap: provenance reaches all_ui_sources (T13, P8, KI1 closed).

[x] No `import mcp` / `provider == "mcp"` outside `providers/mcp/` (grep + verify/mcp.py, P6).



Next slices (Phase 2 #60):

Approved slice order:
```
Slice 2 / T15 — integrations/search v1 + Integrations palette section (composer_mention_search)

Slice 3 / T16 — INSPECT cap steps (pure builders + retrieval_inspector extension)

Slice 4 / T17 — KnowledgePreset.capabilities field + @[tool:user:…] alias resolver

Slice 5 / T18 — Sources UI source_capability label (prestige_dialog)
```

Constraints (must hold):

- MCP is a provider only; no raw MCP tools exposed as primary UX.

- No `import mcp` / `provider == "mcp"` outside `providers/mcp/` (P6).

- Preserve P1-P8; every result carries `cap:` provenance (P4/P8).

- Nothing defaults to write/destructive; unknown classification is default-deny + review (P7).



Test requirements:

See `.cursor/starfall/test-plan.md` (Phase 1 COMPLETE T1–T13) and
`.cursor/starfall/test-plan-phase2.md` (T14–T17 complete; T18 planned).



Open questions blocking handoff:

None. Q1 answered (Option A scoped). Q2 (Phase 4 scope) is roadmap-only.
See `.cursor/starfall/open-questions.md`.

