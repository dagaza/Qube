# Starfall — Open Questions

Unresolved design questions that need a human decision. Move resolved items into
`decisions.md`. Newest at the bottom.

Format:
```
## <id> <question>  [open|answered]
Why it matters: <impact on the design>
Options: <A / B / C>
Blocking: <what phase/gate this blocks, if any>
```

## Q1 `@[tool:user:…]` back-compat scope  [answered]
Why it matters: The URN unifies tokens under `@[cap:…]`; how long do we keep the legacy
alias, and does it resolve to a preset of `cap:` ids?
Options: (A) permanent alias, (B) deprecate after Phase 2, (C) drop pre-GA.
Blocking: Phase 2 composer grammar.
Resolution: Option A (scoped) — see decisions.md 2026-07-23 entry. `@[cap:…]` canonical for
Integrations caps; `@[tool:user:…]` permanent alias expanding to preset cap bundles; built-in
`@[tool:library|internet|…]` unchanged for Phase 2.

## Q2 Phase 4 exact scope  [open]
Why it matters: #62 is labeled hardening/GA but isn't itemized in the plan's four phases.
Options: (A) hardening only, (B) hardening + enterprise gateway provider, (C) TBD.
Blocking: roadmap accuracy.
