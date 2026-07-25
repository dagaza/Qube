# Test Plan — Phase 4 (#62)

STATUS: COMPLETE

Hardening / GA readiness cases. Gate 4 reads Phase 0–3 plans separately; this file
covers T22–T26 only.

## Cases (Phase 4 — #62 hardening)
| # | Area | Case | Test | Status |
|---|------|------|------|--------|
| T22 | KI2 adapter id | NormalizedHit `_adapter` uses namespace short id; full URN in `_capability` | `tests/test_capability_hardening_phase4.py::TestKI2AdapterShortId` + `tests/test_capability_model.py` | pass |
| T23 | KI4 partial deny | Preset bundle partial/full deny summary lists per-cap reasons | `tests/test_capability_hardening_phase4.py::TestKI4PresetPartialDeny` | pass |
| T24 | Denied-path trace | Denied capability invoke persists attachment+invoke INSPECT steps | `tests/test_capability_hardening_phase4.py::TestDeniedPathTrace` | pass |
| T25 | Cited-step wiring | Post-answer cited ids append cited INSPECT step | `tests/test_capability_hardening_phase4.py::TestCitedStepWiring` | pass |
| T26 | Router suggestions | Opt-in suggestions helper is P6-clean; default empty without cache | `tests/test_capability_hardening_phase4.py::TestRouterSuggestions` | pass |

## Regression
- Phase 3: `tests/test_agent_scope_egress_phase3.py`
- Phase 2: `.cursor/starfall/test-plan-phase2.md` cases
- P6 guardrail: `.cursor/starfall/verify/mcp.py`
