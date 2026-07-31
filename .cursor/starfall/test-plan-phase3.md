# Starfall Test Plan — Phase 3 (#61)

Agent scope + egress summary. STATUS: COMPLETE

| ID | Scope | Command / file | Status |
|----|-------|----------------|--------|
| T19 | Agent scope boundaries + enforcement hooks | `tests/test_agent_scope_egress_phase3.py::TestAgentScope` | pass |
| T20 | Step approval for write/destructive + invoke gate + dry-run preview | `tests/test_agent_scope_egress_phase3.py::TestStepApproval` + `TestSessionEgressAndInvoke` | pass |
| T21 | Session egress ledger + summary formatters + P6 on new modules | `tests/test_agent_scope_egress_phase3.py::TestSessionEgressAndInvoke` + `TestPhase3P6Guardrail` + `TestComposerGate` | pass |

Regression (Phase 2): `tests/test_composer_capability_tokens.py`, `tests/test_capability_inspect_steps.py`, `tests/test_preset_capability_alias.py`, `tests/test_sources_capability_provenance_ui.py`.
