# Test Plan

STATUS: COMPLETE

Living test plan for the current initiative. Gate 4 (Tests) reads from here. Flip STATUS
to `COMPLETE` only when every listed case is implemented and passing.

Phase 0 result: 43 tests pass (`pytest tests/test_capability_*.py`), 2026-07-21
(37 foundation + 6 hardening after the Phase 0 diff review).

Provider slice result: +32 tests pass (`pytest tests/test_mcp_provider_client.py
tests/test_capability_persistence.py tests/test_capability_bundle_wiring.py`), 2026-07-22
(T8 provider client/connector, T9 persistence/consent/drift, T10 bundle provenance wiring).

## Strategy
- Provider contract tests run against a **mock MCP server** fixture (no network, no real
  server) so they are deterministic and CI-safe.
- No test may require a live provider or leak `provider == "mcp"` assumptions into
  provider-agnostic layers.

## Cases (Phase 0)
| # | Area | Case | Test | Status |
|---|------|------|------|--------|
| T1 | URN | parse/build/round-trip `cap:<provider>:<ns>/<action>[@version]`; reject malformed | `tests/test_capability_urn.py` | pass |
| T2 | Model | `fingerprint_descriptors` is stable + order-independent; changes on schema/tier drift | `tests/test_capability_model.py::TestFingerprint` | pass |
| T3 | Model | `NormalizedHit.to_evidence_dict` preserves `source_cap` (P8) | `tests/test_capability_model.py::TestNormalizedHitProvenance` | pass |
| T4 | Protocol | a mock provider satisfies `CapabilityProvider` (discover/invoke/health/fingerprint) | `tests/test_capability_protocol.py` | pass |
| T5 | Tier | `escalates_over` ordering read < write < destructive | `tests/test_capability_model.py::TestCapabilityTier` | pass |
| T6 | Mapper | verb heuristics; unknown verb -> DESTRUCTIVE + needs_review (P7); manifest override; slugified URNs valid | `tests/test_capability_mapper.py` | pass |
| T7 | Mapper hardening | URN-collision disambiguation + needs_review (M1); camelCase/snake/dotted slug consistency (L1); un-sluggable namespace raises `CapabilityMappingError` (L2); fingerprint robust to non-JSON schema (L3) | `tests/test_capability_mapper.py`, `tests/test_capability_model.py::TestFingerprint` | pass |

## Cases (provider slice — #58 continuation)
| # | Area | Case | Test | Status |
|---|------|------|------|--------|
| T8 | Provider client | isinstance CapabilityProvider; handshake order (initialize->tools/list, initialized notification); tier mapping; invoke provenance + raw_ref routing; foreign/unknown URN reject; dry_run write no-side-effect; timeout->CapabilityInvocationError; health OK/DOWN; real stdio handshake/invoke/timeout; connector delegates (read allowed, write denied, test_connection) | `tests/test_mcp_provider_client.py` | pass |
| T9 | Persistence/consent | integrations_dir under user_data_root; descriptor cache roundtrip; consent grant/deny persists separately + survives reload; evaluate_access default-deny/allow/needs-review/contract-drift/tier-escalation (P3/P7) | `tests/test_capability_persistence.py` | pass |
| T10 | Bundle wiring (P8/KI1) | cap: URN survives NormalizedHit -> to_evidence_dict -> generic bundle raw_metadata -> evidence_to_ui_source/bundle_to_ui_sources; non-cap rows carry no capability key | `tests/test_capability_bundle_wiring.py` | pass |

## Cases (Phase 1 — #59: Integrations UI + permission model)
| # | Area | Case | Test | Status |
|---|------|------|------|--------|
| T11 | Provider registry | register/get/create/list by normalized string id; unknown id raises UnknownCapabilityProvider; empty-id/non-callable rejected; built-in `mcp` resolves by id to an isinstance CapabilityProvider; reset-then-reload re-registers builtins; registry core + composition root are P6-regex clean (P5/P6) | `tests/test_capability_registry.py` | pass |
| T12 | Consent controller | controller lists descriptor groups + tiers + needs_review; per-capability state derives from `evaluate_access` (not grant presence); grant()/deny() write the exact descriptor and are honored on reload; needs_review stays un-grantable; drift/tier-escalation surfaces re-review (P3/P7) | `tests/test_integrations_consent_controller.py` | pass |
| T13 | LLMWorker main-path (P8/KI1) | when a turn evidence bundle exists, `all_ui_sources` carries `source_capability` for cap: hits via `bundle_to_ui_sources`; mem/rag + web citation ids stay unique (renumbered); non-cap rows carry no capability key | `tests/test_llmworker_ui_sources.py` | pass |

## Acceptance
- All cases pass; no provider-specific import outside `providers/mcp/` (verified by grep). ✓
- Post-review hardening (M1, L1-L3) landed with regression tests. ✓
- Provider slice: mock-server tests deterministic on Windows (sys.executable launch); consent
  is default-deny + drift-aware; cap: provenance reaches the UI on the canonical bundle path. ✓

Phase 2 (#60) planned cases (T14–T18) live in `.cursor/starfall/test-plan-phase2.md` — not
included here while Phase 1 Gate 4 is COMPLETE (verifier regex-scans this file for pytest paths).
