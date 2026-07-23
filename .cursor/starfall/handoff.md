# Implementation Handoff



STATUS: READY

Phase 1 (#59) complete on this branch (see Delivered section below).
Phase 2 (#60): McpConnector provider-delegation + evaluate_access landed in
`mcp_connector.py`; cap-token spine (T14 / slice 1) was built in Run 002 but is
**not on disk** (reverted before commit — see KI3). Next run starts at Phase 2
slice 1 (T14) or palette (T15).



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



Partial (Phase 2 / #60 — on disk in this commit):

```

core/knowledge/connectors/mcp_connector.py

  Delegates to McpCapabilityProvider (persistent session); evaluate_access gate on invoke.

```

Not on disk (Run 002 slice 1 — reverted; rebuild in next Starfall run):

```

core/integrations/capability_invoke.py

core/composer_attachments.py (cap token kind/route)

workers/llm_worker.py (CAPABILITY branch)

tests/test_composer_capability_tokens.py (T14)

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
Slice 1 / T14 — cap token spine + invoke gate (rebuild — not on disk)

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
`.cursor/starfall/test-plan-phase2.md` (T14 planned; T15–T18 planned).



Open questions blocking handoff:

None. Q1 answered (Option A scoped). Q2 (Phase 4 scope) is roadmap-only.
See `.cursor/starfall/open-questions.md`.

