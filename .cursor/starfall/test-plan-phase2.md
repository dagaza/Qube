# Test Plan — Phase 2 (#60)

STATUS: IN PROGRESS (plan only; not scanned by starfall_verify)

Living test plan for Phase 2. Gate 4 for Phase 2 reads from here once Code starts.
Do not merge into `test-plan.md` until T14–T18 are implemented and passing.

## Cases (Phase 2 — Composer palette + presets + INSPECT)

| #   | Area                    | Case                                                                              | Test file (planned)                            | Status  |
| --- | ----------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------- | ------- |
| T14 | Cap token + invoke gate | `@[cap:…]` parse/route; evaluate_access gate; connector consent aligned; P6 clean | tests/test_composer_capability_tokens.py       | planned |
| T15 | Palette search          | Integrations section; fuzzy search; tier/lock from evaluate_access                | tests/test_integrations_capability_search.py   | planned |
| T16 | INSPECT steps           | attachment→invoke→rank→cite pure builders + trace serialization                   | tests/test_capability_inspect_steps.py         | planned |
| T17 | Preset alias            | `@[tool:user:…]` resolves to cap bundle; adapter-only presets unchanged           | tests/test_preset_capability_alias.py          | planned |
| T18 | Sources UI              | prestige/Sources shows source_capability when present                             | tests/test_sources_capability_provenance_ui.py | planned |
