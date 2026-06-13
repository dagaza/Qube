# ADR 001: Skills remain post-routing; sidecar does not replace CognitiveRouterV4

**Status:** Accepted  
**Date:** 2026-06-13  
**Deciders:** Qube maintainers (documented after external architecture review)

## Context

Qube has two related but distinct subsystems:

1. **Tool routing** — where to look before the main model answers: `CognitiveRouterV4`, pre/post-router overrides, web intent (`manual_web` / `live_web`), HYBRID execution, post-retrieval downgrade.
2. **Reasoning skills** — how the model should structure its thinking: compositional prompt injection after routing and retrieval (`core/skills/`), optional `@` palette forcing.

An external proposal suggested **repositioning Skills earlier** in the pipeline and using the **1.7B sidecar** (auxiliary cognition model) as a primary **skill + tool router**, potentially absorbing or replacing `CognitiveRouterV4` and making skills the main abstraction for WEB / MEMORY / RAG / tools.

This ADR records the outcome of stress-testing that proposal against the real codebase (`workers/llm_worker.py`, `mcp/cognitive_router.py`, `core/memory_filters.py`, `core/skills/`, `docs/sidecar_tasks.md`).

### Current turn order (authoritative)

```
User input
  → Discourse grounding (follow-up / query expansion for search only)
  → Pre-router overrides (explicit remember, file search, narrative, composer @*)
  → CognitiveRouterV4.route()
  → Post-router overrides (recall→HYBRID, discourse downgrade, custom RAG triggers,
     manual_web / live_web / auto_web, proactive web veto, discourse web veto)
  → Tool execution (memory / RAG / web per execution_route)
  → Relevance gates
  → Post-retrieval downgrade (empty sources → NONE for prompt build)
  → Skills (activate_skills → skill_guidance in build_prompt_blocks)
  → Main LLM
```

Skills run **after** post-retrieval downgrade because `SkillContext` depends on final `execution_route` and `has_retrieval_sources`. Several builtins use those fields in scoring (e.g. `writing_assistance` boosts on `NONE`; `research_synthesis` boosts when sources exist).

The sidecar is **assistive only** today: query rewrite, source digest, memory background tasks. Its `recommended_target` field on `query_rewrite` is **telemetry only** and must not change `execution_route` (`core/sidecar_prompts.py`, `docs/sidecar_tasks.md`).

## Decision

**Reject** making Skills (or a 1.7B sidecar skill router) the **primary pre-routing decision layer** for tool selection.

**Accept** the following boundaries:

| Layer | Responsibility | Single source of truth? |
|-------|----------------|-------------------------|
| Tier 0 — Hard overrides | Composer attachments, explicit remember, file search, narrative, hard web commands, capability gates | Yes — always wins |
| Tier 1 — Tool routing | `CognitiveRouterV4` + deterministic post-router patches + web intent rules | Yes — `execution_route` |
| Tier 2 — Retrieval + downgrade | Run tools, apply relevance gates, downgrade empty retrieval to `NONE` for prompts | Yes — post-gate route |
| Tier 3 — Reasoning skills | Prompt scaffolding only; max 3 skills, char budget | No authority over tools |
| Sidecar | Rewrite, digest, background memory jobs; optional **advisory** skill hints in future | Never authoritative for routes |

**HYBRID** remains an **execution mode** (memory + RAG, web only when live-web intent is explicit), not a skill.

**Web intent** remains **routing metadata** (`core/memory_filters.py`), not a skill ID — though skills may assume web sources exist once route is WEB.

**Default product posture:** auto-detected skills **off** by default; `@[skill:…]` forcing always works. See Settings → AI & Models → Reasoning skills.

### Conflict hierarchy (strict)

When subsystems disagree, resolve in this order:

1. Composer / explicit user commands (`@internet`, `@file`, hard web verbs)
2. Capability vetoes (internet tool disabled, empty-results policies)
3. Discourse safety rules (ungrounded web veto; follow-up downgrade)
4. `CognitiveRouterV4` + `memory_filters` web intent (`manual_web`, `live_web`, temporal carve-outs)
5. Post-retrieval downgrade (prompt-time route only)
6. Skills (prompt shaping only)
7. Sidecar (assistive; telemetry at most)

## Alternatives considered

### A. Skills pre-routing as primary tool router

**Rejected.** Skills answer “how to think,” not “where to look.” Moving them earlier:

- Breaks `SkillContext` dependencies on `execution_route` and `has_retrieval_sources`.
- Fights post-retrieval downgrade (tools run, then route may change for prompt build).
- Duplicates web intent logic already split across router substrings, regex topics, and temporal guards (`test_web_intent_split`).
- Creates dual authority with `CognitiveRouterV4` and post-router overrides.

### B. 1.7B sidecar replaces CognitiveRouterV4

**Rejected for now.** Partially feasible on obvious lanes only. V4 adds drift detection, dynamic thresholds, latency/RAG load control, tier-2 ambiguity→HYBRID, tier-3 adaptive calibration, and embedding-vs-chat margins. A 1.7B JSON classifier cannot replicate hardened edge cases (e.g. `"schedule my tasks for today"` → NONE vs weather → WEB) without retaining the full rule stack — at which point the model adds latency without replacing the rules.

### C. Sidecar post-route skill suggestion (boost only)

**Accepted as future option (Phase 1–2).** A new `SidecarTask` could propose `skill_ids` **after** `execution_route` is final, merging with trigger/embedding scores in `activate_skills` — same pattern as optional embedding boost. Must not emit binding tool flags.

### D. Skills always on by default

**Deferred.** Auto-detect remains off until eval/manual QA show predictable activation and acceptable prompt cost. Forced `@` skills and Settings toggle remain the discovery paths.

## Consequences

### Positive

- Routing regressions stay isolated: `tests/test_skills_router_non_regression.py`, `test_web_intent_split`, `test_web_veto_fallback`, router-eval harness.
- Clear ownership: router team vs skills team vs sidecar tasks.
- Post-retrieval downgrade and web veto fixes remain valid (no “WEB prompt with no sources” from skill-router mistakes).
- Skills can evolve (new builtins, centroids, palette) without touching lane logic.

### Negative / constraints

- Users must enable auto-detect skills or use `@` — skills are not “free” scaffolding for all turns.
- Two classifiers coexist in the product mental model (router lanes vs reasoning skills); documentation must keep the distinction explicit.
- External agents may propose merging layers; this ADR is the reference for why that is costly.

### Implementation guardrails

- `core/skills/` must **not** import `cognitive_router` or `memory_filters` (enforced in tests).
- `activate_skills()` stays in `llm_worker` **after** §2.75 post-retrieval downgrade.
- Sidecar prompts must continue to state that routing fields are read-only context.
- Any future sidecar skill task must fail closed to rule-based `activate_skills` on timeout/parse error.

## Failure modes if decision is violated

| Failure | Severity | Example |
|---------|----------|---------|
| Skill router WEB vs proactive veto | Critical | Internet off; WEB prompt; hallucinated `[W]` citations |
| Skill router retrieval vs discourse downgrade | Critical | Follow-up answerable from thread; wasted retrieval |
| `manual_web` overridden by model | Critical | User said “search the web”; no search runs |
| HYBRID web mis-gating | High | Web always or never on hybrid turns |
| Temporal false positive | High | Personal “today” planner routed to WEB |
| Double classification latency | High | Sidecar route JSON + V4 + skills before tools |
| Skill guidance vs empty sources | Medium | `research_synthesis` active after downgrade to NONE |

## Phased evolution (optional, within this ADR)

| Phase | Routing authority | Skills | Sidecar | Risk |
|-------|-------------------|--------|---------|------|
| **1 — Coexist** | Unchanged | Post-route, rule-based | Log advisory skill suggestions vs actual activations | Low |
| **2 — Advisory boost** | Unchanged | Merge sidecar skill boosts above score floor | No `recommended_target` → route changes | Medium |
| **3 — Not approved** | V4 + rules remain unless full parity proven | Still post-route | Never primary router without re-homing vetoes/downgrade | High |

Phase 3 must not proceed without passing existing web-intent and router-eval baselines.

## References

- [`docs/cognitive_router.md`](../cognitive_router.md) — end-to-end routing flow
- [`docs/sidecar_tasks.md`](../sidecar_tasks.md) — sidecar constraints and tasks
- [`docs/skills_manual_qa.md`](../skills_manual_qa.md) — manual QA for skills layer
- `workers/llm_worker.py` — orchestration, §2.75 downgrade, skills hook ~§3
- `mcp/cognitive_router.py` — `CognitiveRouterV4`
- `core/memory_filters.py` — `query_implies_live_web_intent`, `should_run_internet_search_for_route`
- `core/skills/activation.py` — `activate_skills`, skip reasons
- `tests/test_skills_router_non_regression.py` — orthogonality contract
